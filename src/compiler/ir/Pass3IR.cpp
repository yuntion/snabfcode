#include "Pass3IR.h"
#include "Pass2IR.h"
#include "pass3/ConcreteVerifier.h"
#include "pass3/CostProfile.h"
#include "pass3/ParameterMILP.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <map>
#include <set>
namespace snabf::ir {
namespace p = ::snabf::pass3;
namespace {
using O = p::Op;
int64_t num(DictionaryAttr d, StringRef k) {
  return field<IntegerAttr>(d, k).getInt();
}
double real(DictionaryAttr d, StringRef k) {
  return field<FloatAttr>(d, k).getValueAsDouble();
}
struct Input {
  p::Graph g;
  ProfileOp profile, target;
  SmallVector<ConvertOp> bridges;
  llvm::DenseMap<Value, int> results;
  ArrayAttr keys;
};
ProfileOp domain(Value v) {
  auto t = cast<CtType>(v.getType());
  auto result = v.getDefiningOp();
  auto p = result ? SymbolTable::lookupNearestSymbolFrom<ProfileOp>(
                        result, t.getProfile())
                  : ProfileOp();
  require(p && t.getN() == int64_t(p.getRing()) && t.getLevel() >= 0 &&
              t.getLevel() < int64_t(p.getQ().size()),
          "unbound_ciphertext_key_domain");
  return p;
}
Input extract(ModuleOp m, p::IRMagnitudeOrigins *origins = nullptr) {
  Input in;
  auto &g = in.g;
  Builder b(m.getContext());
  if (origins) {
    auto mode = m->getAttrOfType<StringAttr>("snabf.interval_mode");
    origins->intervalMode = mode ? mode.str() : "naive";
  }
  m.walk([&](SourceOp s) {
    require(!in.profile, "multiple_source_domains_not_emittable");
    in.profile = domain(s.getOut());
  });
  require(in.profile && !in.profile->hasAttr("snabf.target_epoch"),
          "missing_source_key_domain");
  m.walk([&](ConvertOp c) {
    require(in.bridges.size() < 2 && (!in.target || in.target == domain(c.getOut())),
            "multiple_or_incompatible_cut_domains");
    in.bridges.push_back(c);
    in.target = domain(c.getOut());
    int64_t l = c.getOut().getType().getLevel();
    require(domain(c.getInput()) == in.profile && in.target != in.profile &&
                in.target->hasAttr("snabf.target_epoch") &&
                in.target.getRing() < in.profile.getRing() &&
                in.profile.getRing() % in.target.getRing() == 0 &&
                c.getInput().getType().getLevel() >= l &&
                in.target.getQ() == in.profile.getQ().take_front(l + 1) &&
                in.target.getP() == in.profile.getP() &&
                in.target.getKeyId() != in.profile.getKeyId(),
            "incompatible_switching_key_domains");
  });
  m.walk([&](ProfileOp f) {
    require(f == in.profile || f == in.target, "unaccounted_key_domain");
  });
  g.n = in.profile.getRing();
  g.levels = in.profile.getQ().size();
  in.keys =
      field<ArrayAttr>(field<DictionaryAttr>(m, "snabf.inventory"), "keys");
  auto &edges = in.results;
  int polyIndex = 0, stepIndex = 0, downIndex = 0;
  auto make = [&](std::string id, O op, int l, std::vector<int> a,
                  std::vector<double> w, double mag, int mask = -1,
                  int poly = -1, int node = -1) {
    int i = g.edges.size();
    g.edges.push_back({std::move(id), op, l, std::move(a), std::move(w), mag,
                       mask, poly, node});
    return i;
  };
  auto magnitude = [&](int a) { return g.edges.at(a).magnitude; };
  auto at = [&](Value v) {
    auto it = edges.find(v);
    require(it != edges.end(),
            "Pass 3 requires a topological encrypted SSA region");
    return it->second;
  };
  m.walk([&](Operation *o) {
    if (auto sink = dyn_cast<SinkOp>(o)) {
      require(g.sink == -1, "multiple sinks outside this Pass 3 family");
      require(domain(sink.getInput()) == (in.target ? in.target : in.profile),
              "sink_key_domain_mismatch");
      g.sink = at(sink.getInput());
      g.approximation =
          real(field<BudgetAttr>(o, "budget").getFields(), "approx_error");
      return;
    }
    if (!o->getNumResults() || !isa<CtType>(o->getResult(0).getType()))
      return;
    auto out = o->getResult(0);
    auto outDomain = domain(out);
    require(outDomain == in.profile || outDomain == in.target,
            "unaccounted_key_domain");
    if (!isa<ConvertOp>(o))
      for (Value input : o->getOperands())
        if (isa<CtType>(input.getType()))
          require(domain(input) == outDomain, "implicit_cross_domain_operand");
    int l = cast<CtType>(out.getType()).getLevel();
    int result = -1;
    auto first = g.edges.size();
    if (auto c = dyn_cast<ConvertOp>(o)) {
      int a = at(c.getInput());
      std::string id = in.bridges.size() == 1 ? "ring_down" : "ring_down." + std::to_string(downIndex++);
      if (g.edges[a].level != l)
        a = make(id + ".align", O::Drop, l, {a}, {1}, magnitude(a));
      result = make(id, O::Down, l, {a}, {1}, magnitude(a));
    } else if (auto s = dyn_cast<SourceOp>(o)) {
      result = make("input", O::Source, l, {}, {},
                    field<FloatAttr>(o, "input_bound").getValueAsDouble());
    } else if (auto op = dyn_cast<LintransOp>(o)) {
      int a = at(op.getInput()), level = g.edges[a].level;
      auto route = field<DictionaryAttr>(op, "selected");
      auto ts = field<DenseI64ArrayAttr>(route, "terms");
      auto wt = preparedWeights(op.getTable(), op.getDiagonals());
      auto shape = wt.getType().getShape();
      require(shape.size() == 2 && shape[0] * 3 == ts.size() && level == l + 1,
              "mismatched tower/level");
      std::vector<double> maxRow(shape[0]);
      size_t j = 0;
      for (auto f : wt.getValues<APFloat>()) {
        auto &v = maxRow[j++ / shape[1]];
        v = std::max(v, std::abs(f.convertToDouble()));
      }
      int site = field<IntegerAttr>(o, "site").getInt();
      std::string prefix = "lin" + std::to_string(site) + ".";
      std::map<int, int> babies{{0, a}};
      std::map<int, std::vector<int>> groups;
      std::map<int, double> weights;
      for (int t = 0; t < ts.size(); t += 3) {
        int baby = ts[t + 1], giant = ts[t + 2];
        if (!babies.contains(baby)) {
          babies[baby] = make(prefix + "baby" + std::to_string(baby), O::Baby,
                              level, {a}, {1}, magnitude(a));
          g.edges[babies[baby]].shift = baby;
          g.edges[babies[baby]].routeSite = site;
        }
        int parent = babies[baby];
        double w = maxRow[t / 3];
        int v = make(prefix + "mask" + std::to_string(ts[t]), O::Mask, level,
                     {parent}, {w}, w * magnitude(a), level);
        groups[giant].push_back(v);
      }
      auto sum = [&](std::string id, std::vector<int> v) {
        if (v.size() == 1)
          return v[0];
        double mag = 0;
        for (int i : v)
          mag += magnitude(i);
        return make(prefix + id, O::Add, level, v,
                    std::vector<double>(v.size(), 1), mag);
      };
      std::vector<int> all;
      for (auto &[giant, terms] : groups) {
        int v = sum("coset" + std::to_string(giant), terms);
        if (giant) {
          v = make(prefix + "giant" + std::to_string(giant), O::Giant, level,
                   {v}, {1}, magnitude(v));
          g.edges[v].shift = giant;
          g.edges[v].routeSite = site;
        }
        all.push_back(v);
      }
      int v = sum("merge", all);
      result = make(prefix + "rescale", O::Rescale, l, {v}, {1}, magnitude(v));
    } else if (auto op = dyn_cast<PolyEvalOp>(o)) {
      auto sel = field<DictionaryAttr>(o, "poly_selected");
      int a = at(op.getInput()), level = g.edges[a].level;
      auto tr = transferRecord(op, sel, level);
      std::set<int> relins;
      for (auto it : field<ArrayAttr>(tr, "sites")) {
        auto d = cast<DictionaryAttr>(it);
        if (field<StringAttr>(d, "primitive").getValue() == "relin")
          relins.insert(num(d, "node"));
      }
      p::Poly poly{level, int(num(sel, "output")), {}};
      std::vector<int> values;
      int pi = polyIndex++;
      for (auto item : field<ArrayAttr>(sel, "nodes")) {
        auto d = cast<DictionaryAttr>(item);
        auto opName = field<StringAttr>(d, "op").str();
        int node = values.size(), depth = num(d, "depth"), end = level - depth;
        bool mul = opName == "mul", linear = opName == "linear";
        int start = end + (mul || linear);
        std::string id =
            "poly" + std::to_string(pi) + ".n" + std::to_string(node) + ".";
        std::vector<int> args, refs;
        for (int idx : field<DenseI64ArrayAttr>(d, "args").asArrayRef()) {
          refs.push_back(idx);
          int v = values.at(idx);
          if (g.edges[v].level != start)
            v = make(id + "drop" + std::to_string(args.size()), O::Drop, start,
                     {v}, {1}, magnitude(v));
          args.push_back(v);
        }
        poly.nodes.push_back({opName, refs, depth});
        int v = a;
        if (mul) {
          v = make(id + "mul", O::Multiply, start, args,
                   {magnitude(args[1]), magnitude(args[0])},
                   magnitude(args[0]) * magnitude(args[1]));
          require(relins.erase(node) == 1, "missing Pass 2 relin contract");
          v = make(id + "relin", O::Relin, start, {v}, {1}, magnitude(v));
        } else if (linear || opName == "add") {
          std::vector<int> terms;
          double mag = 0;
          auto w = field<DenseF64ArrayAttr>(d, "weights");
          for (int j = 0; j < int(args.size()); ++j) {
            int t = args[j];
            if (linear) {
              double c = std::abs(w[j]);
              t = make(id + "pt" + std::to_string(j), O::Coefficient, start,
                       {t}, {c}, magnitude(t) * c, -1, pi, node);
              if (origins) origins->polynomialLiterals.emplace(t, w[j]);
            }
            terms.push_back(t);
            mag += magnitude(t);
          }
          v = terms.size() == 1
                  ? terms[0]
                  : make(id + "add", O::Add, start, terms,
                         std::vector<double>(terms.size(), 1), mag);
        } else
          require(opName == "input" && node == 0, "unknown polynomial node");
        if (mul || linear)
          v = make(id + "rescale", O::Rescale, end, {v}, {1}, magnitude(v));
        double signedConstant = real(d, "constant");
        double constant = std::abs(signedConstant);
        if (constant) {
          v = make(id + "constant", O::AddPlain, end, {v}, {1},
                   magnitude(v) + constant);
          if (origins) origins->polynomialLiterals.emplace(v, signedConstant);
        }
        values.push_back(v);
      }
      require(relins.empty(), "unbound Pass 2 relin sites");
      result = values.at(poly.output);
      g.polys.push_back(std::move(poly));
    } else if (auto s = dyn_cast<StepOp>(o)) {
      auto k = s.getKind();
      int a = at(o->getOperand(0));
      std::string id = "step" + std::to_string(stepIndex++);
      if (k == "ckks.add" || k == "ckks.mul") {
        int c = at(o->getOperand(1));
        bool mul = k == "ckks.mul";
        for (int *v : {&a, &c})
          if (g.edges[*v].level != l)
            *v = make(id + ".drop" + std::to_string(*v), O::Drop, l, {*v}, {1},
                      magnitude(*v));
        result = make(id, mul ? O::Multiply : O::Add, l, {a, c},
                      mul ? std::vector<double>{magnitude(c), magnitude(a)}
                          : std::vector<double>{1, 1},
                      mul ? magnitude(a) * magnitude(c)
                          : magnitude(a) + magnitude(c));
      } else if (k == "ckks.mul_plain") {
        double w = linearNorm(plainWeights(o->getOperand(1)), false);
        result = make(id, O::Coefficient, l, {a}, {w}, w * magnitude(a));
      } else {
        require(k == "ckks.level_reduce" || k == "ckks.rescale" ||
                    k == "ckks.relinearize",
                "unsupported Pass 3 step");
        result = make(id,
                      k == "ckks.level_reduce" ? O::Drop
                      : k == "ckks.rescale"    ? O::Rescale
                                               : O::Relin,
                      l, {a}, {1}, magnitude(a));
      }
    } else
      throw ContractError("Pass 3: unsupported encrypted operation " +
                          name(o).str());
    int n = isa<ConvertOp>(o) ? cast<CtType>(o->getOperand(0).getType()).getN()
                              : cast<CtType>(out.getType()).getN();
    for (size_t i = first; i < g.edges.size(); ++i)
      g.edges[i].ring = n;
    require(result >= 0 && g.edges.at(result).level == l &&
                edges.try_emplace(out, result).second,
            "unbound_concrete_ssa_result");
  });
  return in;
}

DictionaryAttr options(Builder &b, const p::Options &o) {
  NamedAttrList d;
  auto i = [&](StringRef k, int v) { d.set(k, b.getI64IntegerAttr(v)); };
  auto f = [&](StringRef k, double v) { d.set(k, b.getF64FloatAttr(v)); };
  i("min_scale", o.minScale);
  i("max_scale", o.maxScale);
  i("max_k", o.maxK);
  if (o.blockSearch()) {
    i("gadget_alpha", o.alpha);
    d.set("heterogeneous_scales", b.getBoolAttr(o.heterogeneous));
    d.set("heir_scale_anchor", b.getBoolAttr(o.heirScaleAnchor));
    if (o.customLimits()) {
      f("prime_bw_min", o.primeWidthMin); f("prime_bw_max", o.primeWidthMax);
      i("aux_prime_margin_bits", o.auxMarginBits);
    }
  }
  f("tolerance", o.tolerance);
  f("security_cap", o.securityCap);
  f("min_base", o.minBase);
  f("guard", o.guardBits);
  if (o.costProfile) {
    d.set("cost_mode",b.getStringAttr("calibrated-sites-v1"));
    d.set("cost_profile_sha256",b.getStringAttr(o.costProfile->digest()));
    d.set("cost_hardware_id",b.getStringAttr(o.costProfile->hardwareID()));
  }
  return d.getDictionary(b.getContext());
}
p::Options readOptions(DictionaryAttr d, ModuleOp m) {
  p::Options o{int(num(d, "min_scale")), int(num(d, "max_scale")),
          int(num(d, "max_k")),     real(d, "tolerance"),
          real(d, "security_cap"),  real(d, "min_base"),
          real(d, "guard")};
  if (d.contains("gadget_alpha")) {
    o.alpha=int(num(d,"gadget_alpha"));
    o.heterogeneous=field<BoolAttr>(d,"heterogeneous_scales").getValue();
    o.heirScaleAnchor=field<BoolAttr>(d,"heir_scale_anchor").getValue();
    if (d.contains("prime_bw_min") || d.contains("prime_bw_max") || d.contains("aux_prime_margin_bits")) {
      o.primeWidthMin=real(d,"prime_bw_min"); o.primeWidthMax=real(d,"prime_bw_max");
      o.auxMarginBits=int(num(d,"aux_prime_margin_bits"));
    }
  }
  auto profile=m->getAttrOfType<StringAttr>("snabf.gadget_cost_profile");
  if (m->hasAttr("snabf.gadget_cost_profile") || d.contains("cost_mode") || d.contains("cost_profile_sha256") || d.contains("cost_hardware_id")) {
    require(profile && field<StringAttr>(d,"cost_mode").getValue()=="calibrated-sites-v1", "missing_gadget_cost_profile_or_mode");
    o.costProfile=p::CostProfile::parse(profile.str(),field<StringAttr>(d,"cost_hardware_id").str());
    require(o.costProfile->digest()==field<StringAttr>(d,"cost_profile_sha256").str(),"stale_gadget_cost_profile");
  }
  return o;
}
// Conditional bounds in canonical units. Hoeffding's range bound applies to
// the centered error even for dependent sites; no assumption of zero mean or
// independent CRT rows enters the decoded absolute-error recurrence.
void noiseFields(Builder &b, NamedAttrList &a, const p::Graph &g,
                 const p::Solution &s, size_t i) {
  const auto &e = g.edges[i];
  long double product = 1;
  for (int j = 0; j < s.k[i]; ++j)
    product *= s.q[j];
  auto put = [&](StringRef key, long double x) {
    double d = double(x);
    if (d < x)
      d = std::nextafter(d, INFINITY);
    a.set(key, b.getF64FloatAttr(d));
  };
  int n = e.ring ? e.ring : g.n;
  bool sw = p::keyOp(e.op) || e.op == O::Down;
  long double trunc = s.k[i] ? p::residualFactor(e.op, n) * (product - 1) : 0;
  put("truncation_absolute_canonical", trunc);
  put("truncation_centered_proxy", trunc * trunc);
  long double err = s.errors[i] * std::exp2(s.physical[i]);
  put("conditional_mean_abs_canonical", err);
  put("conditional_centered_proxy", err * err);
  put("moddown_or_rescale_rounding",
      e.op == O::Rescale || sw ? p::rounding(n) : 0);
  if (sw) {
    long double key = p::switchKeyError(s.q,s.p,e.level,s.k[i],n);
    put("ordinary_switch_key_error", key);
  }
}
std::string decimal(ArrayRef<int64_t>, ArrayRef<int64_t>);
ArrayAttr keyDomains(Builder &b, Input &in) {
  SmallVector<Attribute> domains;
  auto addBasis = [&](NamedAttrList &d, ArrayRef<int64_t> q,
                      ArrayRef<int64_t> p) {
    d.set("q", b.getDenseI64ArrayAttr(q));
    d.set("p", b.getDenseI64ArrayAttr(p));
    d.set("level_q", b.getI64IntegerAttr(q.size() - 1));
    d.set("level_p", b.getI64IntegerAttr(p.size() - 1));
    d.set("q_modulus", b.getStringAttr(decimal(q, {})));
    d.set("qp_modulus", b.getStringAttr(decimal(q, p)));
  };
  for (auto profile : {in.profile, in.target}) {
    if (!profile)
      continue;
    NamedAttrList d;
    d.set("role", b.getStringAttr(profile == in.profile ? "source" : "target"));
    d.set("key_id", profile.getKeyIdAttr());
    d.set("profile", profile.getSymNameAttr());
    d.set("ring_dim", profile.getRingAttr());
    addBasis(d, profile.getQ(), profile.getP());
    d.set("ciphertext_basis", b.getStringAttr("Q"));
    SmallVector<Attribute> keys;
    auto key = [&](StringRef role, bool published,
                   std::optional<int64_t> g = std::nullopt) {
      NamedAttrList a;
      a.set("role", b.getStringAttr(role));
      a.set("published", b.getBoolAttr(published));
      a.set("basis", b.getStringAttr("QP"));
      if (g)
        a.set("galois", b.getI64IntegerAttr(*g));
      keys.push_back(a.getDictionary(b.getContext()));
    };
    key("secret", false);
    if (profile == in.profile)
      key("public", true);
    if (llvm::any_of(in.g.edges, [&](const auto &e) {
          return e.op == O::Relin && int64_t(profile.getRing()) == e.ring;
        }))
      key("relin", true);
    std::set<int64_t> elements;
    profile->getParentOfType<ModuleOp>().walk([&](LintransOp op) {
      if (domain(op.getInput()) != profile)
        return;
      auto selected = field<DictionaryAttr>(op, "selected");
      for (int64_t g : field<DenseI64ArrayAttr>(selected, "keys").asArrayRef())
        elements.insert(g);
    });
    for (auto g : elements)
      key("galois", true, g);
    d.set("keys", b.getArrayAttr(keys));
    domains.push_back(d.getDictionary(b.getContext()));
  }
  if (!in.bridges.empty()) {
    NamedAttrList d;
    d.set("role", b.getStringAttr("switching_key_ext"));
    d.set("basis", b.getStringAttr("QP"));
    d.set("key_kind", b.getStringAttr("ring_down"));
    d.set("published", b.getBoolAttr(true));
    d.set("source_key", in.profile.getKeyIdAttr());
    d.set("target_key", in.target.getKeyIdAttr());
    // The external product is large-ring arithmetic; the embedded small secret
    // must also be screened.
    d.set("ring_dim", in.profile.getRingAttr());
    d.set("embedded_secret_ring", in.target.getRingAttr());
    addBasis(d, in.target.getQ(), in.target.getP());
    domains.push_back(d.getDictionary(b.getContext()));
  }
  return b.getArrayAttr(domains);
}
DictionaryAttr record(Builder &b, Input &in, const p::Options &o,
                      const p::Solution &s) {
  NamedAttrList d;
  SmallVector<Attribute> edges, instances;
  bool mixed = llvm::any_of(
      in.g.edges, [](const auto &edge) { return edge.op == O::Down; });
  for (size_t i = 0; i < in.g.edges.size(); ++i) {
    auto &e = in.g.edges[i];
    NamedAttrList a;
    a.set("ID", b.getStringAttr(e.id));
    if (mixed) {
      a.set("kernel_ring", b.getI64IntegerAttr(e.ring));
      bool down = e.op == O::Down;
      auto before =
          e.ring == int64_t(in.profile.getRing()) ? in.profile : in.target;
      a.set("input_key_domain", before.getKeyIdAttr());
      a.set("output_key_domain",
            down ? in.target.getKeyIdAttr() : before.getKeyIdAttr());
      a.set("kernel_domain",
            b.getStringAttr(down                   ? "switching_key_ext"
                            : before == in.profile ? "source"
                                                   : "target"));
    }
    a.set("op", b.getStringAttr(p::name(e.op)));
    a.set("Level", b.getI64IntegerAttr(e.level));
    a.set("K", b.getI64IntegerAttr(s.k[i]));
    if (o.blockSearch() && (p::keyOp(e.op) || e.op==O::Down)) {
      SmallVector<int64_t> active;
      for (int j=0; j<=e.level; j+=int(s.p.size()))
        active.push_back(std::min(e.level,j+int(s.p.size())-1)>=s.k[i]);
      a.set("fixed_block_active", b.getDenseI64ArrayAttr(active));
      a.set("planned_full_blocks_skipped", b.getI64IntegerAttr(s.k[i]/s.p.size()));
    }
    a.set("inputs", b.getDenseI64ArrayAttr(SmallVector<int64_t>(
                        e.inputs.begin(), e.inputs.end())));
    a.set("gains", b.getDenseF64ArrayAttr(e.gains));
    a.set("magnitude", b.getF64FloatAttr(e.magnitude));
    a.set("physical_log_scale", b.getF64FloatAttr(s.physical[i]));
    a.set("decoded_arithmetic_bound", b.getF64FloatAttr(s.errors[i]));
    noiseFields(b, a, in.g, s, i);
    edges.push_back(a.getDictionary(b.getContext()));
  }
  d.set("edges", b.getArrayAttr(edges));
  d.set("ring", b.getI64IntegerAttr(in.g.n));
  d.set("q", b.getDenseI64ArrayAttr(s.q));
  d.set("p", b.getDenseI64ArrayAttr(s.p));
  d.set("scale", b.getI64IntegerAttr(s.scale));
  d.set("security_cap", b.getF64FloatAttr(o.securityCap));
  d.set("options", options(b, o));
  if (o.costProfile) {
    SmallVector<Attribute> charges;
    double total=0;
    for (const auto &price:p::replayPrices(in.g,*o.costProfile,s)) {
      NamedAttrList charge;
      charge.set("edges",b.getDenseI64ArrayAttr(SmallVector<int64_t>(price.edges.begin(),price.edges.end())));
      charge.set("median_us",b.getF64FloatAttr(price.us));
      charges.push_back(charge.getDictionary(b.getContext()));total+=price.us;
    }
    d.set("gadget_site_charges",b.getArrayAttr(charges));
    d.set("gadget_objective_us",b.getF64FloatAttr(total));
  }
  d.set("approximation", b.getF64FloatAttr(in.g.approximation));
  d.set("sink", b.getI64IntegerAttr(in.g.sink));
  d.set("key_id", in.profile.getKeyIdAttr());
  d.set("arithmetic_model",
        b.getStringAttr(o.blockSearch() ? "centered-prefix.fixed-blocks.v2; conditional bounded encoding; common ternary-key event"
                        : "centered-prefix.oneP.v1; conditional bounded encoding; common ternary-key event"));
  if (mixed) {
    // Unlike the frozen single-profile Pass 3 ABI, these records are scoped.
    d.set("key_domains", keyDomains(b, in));
  } else {
    for (auto x : {"public_key:QP", "relinearization_key:QP", "ciphertext:Q"})
      instances.push_back(b.getStringAttr(x));
    for (auto k : in.keys)
      instances.push_back(b.getStringAttr(
          "galois:" + std::to_string(num(cast<DictionaryAttr>(k), "galois")) +
          ":QP"));
    d.set("public_instances", b.getArrayAttr(instances));
  }
  return d.getDictionary(b.getContext());
}

std::string decimal(ArrayRef<int64_t> v, ArrayRef<int64_t> p = {}) {
  boost::multiprecision::cpp_int a = 1;
  for (auto x : v)
    a *= x;
  for (auto x : p)
    a *= x;
  return a.str();
}
void checkSecurity(StringRef text, DictionaryAttr plan) {
  auto parsed = llvm::json::parse(text);
  require(bool(parsed), "invalid security evidence JSON");
  auto *obj = parsed->getAsObject();
  require(
      obj &&
          obj->getString("model") ==
              "LWE.estimate.rough/classical ADPS16; GSA uSVP, dual_hybrid" &&
          obj->getString("error") == "discrete Gaussian sigma=3.2 (estimator "
                                     "model for bounded sampler)" &&
          obj->getString("estimator") ==
              "ae45d740ceb138af87a4f92edc592a0ab0d4f43f" &&
          obj->getString("samples") == "unbounded" &&
          obj->getString("secret") == "iid uniform ternary" &&
          obj->getInteger("ring") == num(plan, "ring"),
      "estimator context mismatch");
  auto *instances = obj->getObject("instances");
  auto *inventory = obj->getArray("inventory");
  auto ids = field<ArrayAttr>(plan, "public_instances");
  require(instances && inventory && inventory->size() == ids.size(),
          "incomplete security inventory");
  for (size_t i = 0; i < ids.size(); ++i)
    require((*inventory)[i].getAsString() ==
                cast<StringAttr>(ids[i]).getValue(),
            "stale estimator key inventory");
  auto q = field<DenseI64ArrayAttr>(plan, "q").asArrayRef(),
       p = field<DenseI64ArrayAttr>(plan, "p").asArrayRef();
  if (field<DictionaryAttr>(plan,"options").contains("gadget_alpha")) {
    auto bound=[&](StringRef name,ArrayRef<int64_t> primes){
      auto *a=obj->getArray(name);
      if (!a || a->size()!=primes.size()) return false;
      for(size_t i=0;i<primes.size();++i)
        if((*a)[i].getAsInteger()!=primes[i]) return false;
      return true;
    };
    require(bound("ordered_q",q) && bound("ordered_p",p),"stale_ordered_estimator_basis");
  }
  boost::multiprecision::cpp_int cap = 1;
  cap <<= int(std::ceil(real(plan, "security_cap")));
  for (auto [id, mod] : std::vector<std::pair<std::string, std::string>>{
           {"Q", decimal(q)}, {"QP", decimal(q, p)}, {"envelope", cap.str()}}) {
    auto *x = instances->getObject(id);
    auto *att = x ? x->getObject("attacks_log2_rop") : nullptr;
    auto acceptable = [&](StringRef name) {
      auto v = att ? att->getNumber(name) : std::nullopt;
      return v && std::isfinite(*v) && *v >= 128;
    };
    require(att && x->getString("modulus") == mod && acceptable("usvp") &&
                acceptable("dual_hybrid"),
            "failed or stale concrete security screen");
  }
}
} // namespace
ConcreteSSAContext concreteSSAContext(ModuleOp m, DictionaryAttr par) {
  auto in = extract(m);
  Builder b(m.getContext());
  auto edges = field<ArrayAttr>(par, "edges");
  require(edges.size() == in.g.edges.size(), "stale_concrete_ssa_edges");
  ConcreteSSAContext out;
  out.keyDomains = keyDomains(b, in);
  for (const auto &[v, i] : in.results) {
    auto edge = dyn_cast<DictionaryAttr>(edges[i]);
    require(edge &&
                field<StringAttr>(edge, "ID").getValue() == in.g.edges[i].id &&
                num(edge, "Level") == cast<CtType>(v.getType()).getLevel() &&
                std::isfinite(real(edge, "physical_log_scale")),
            "stale_concrete_ssa_edge");
    out.values.try_emplace(v, edge);
  }
  return out;
}
DictionaryAttr boundaryParameters(ModuleOp m, DictionaryAttr base,
                                  DictionaryAttr check) {
  Builder b(m.getContext());
  auto in = extract(m);
  auto o = readOptions(field<DictionaryAttr>(base, "options"),m);
  p::Solution s;
  s.q.assign(in.profile.getQ().begin(), in.profile.getQ().end());
  s.p.assign(in.profile.getP().begin(), in.profile.getP().end());
  s.scale = in.profile.getDefaultLogScale();
  if (check) {
    for (auto e : field<ArrayAttr>(check, "edges"))
      s.k.push_back(num(cast<DictionaryAttr>(e), "K"));
    auto v = p::replay(in.g, o, s);
    require(v.reason.empty(), v.reason);
  } else
    s = p::optimizeAtBoundary(in.g, o, s);
  auto rec = record(b, in, o, s);
  require(!check || rec == check,
          "stale mixed parameter/scale/gadget transfer");
  return rec;
}
void verifySourceScreen(ModuleOp m, DictionaryAttr base) {
  checkSecurity(field<StringAttr>(m, "snabf.security_screen").getValue(), base);
}
void publishPass3(ModuleOp m, const p::Options &o, bool fixed, StringRef audit,
                  StringRef security, double sourceFraction) {
  verifyPlan(m);
  Builder b(m.getContext());
  auto *trace = fixed ? nullptr : p::rejectionTraceStream();
  p::IRMagnitudeOrigins origins;
  auto in = extract(m, trace ? &origins : nullptr);
  p::Solution s;
  require(sourceFraction > 0 && sourceFraction <= 1 && (!fixed || sourceFraction == 1), "invalid_source_arithmetic_fraction");
  const double requestedFraction = sourceFraction;
  if (!fixed) {
    auto solve = [&](const p::Options &opts) {
      return trace ? p::optimize(in.g, opts, *trace, origins) : p::optimize(in.g, opts);
    };
    auto search = o;
    if (sourceFraction < 1)
      search.tolerance = in.g.approximation + sourceFraction * (o.tolerance - in.g.approximation);
    try {
      s = solve(search);
    } catch (const std::runtime_error &e) {
      // A planning reserve must not reject an otherwise feasible source-only
      // fallback. Never absorb unknown solver errors or admission failures.
      if (sourceFraction == 1 || StringRef(e.what()) != "Pass 3 infeasible under requested scale, precision and QP bounds") throw;
      sourceFraction = 1;
      s = solve(o);
    }
    // This is a search reserve, not evidence about either conversion. Always
    // replay the actual graph under the caller's original acceptance contract.
    if (sourceFraction < 1) { auto v = p::replay(in.g, o, s); require(v.reason.empty(), v.reason); }
  }
  else {
    s.scale = in.profile.getDefaultLogScale();
    s.q.assign(in.profile.getQ().begin(), in.profile.getQ().end());
    s.p.assign(in.profile.getP().begin(), in.profile.getP().end());
    s.k.resize(in.g.edges.size());
    auto v = p::replay(in.g, o, s);
    require(v.reason.empty(), v.reason);
  }
  require(o.costProfile || !m->hasAttr("snabf.gadget_cost_profile"),"unexpected_gadget_cost_profile_in_legacy_mode");
  if (o.costProfile) m->setAttr("snabf.gadget_cost_profile",b.getStringAttr(o.costProfile->json()));
  auto r = record(b, in, o, s);
  m->setAttr("snabf.pass3", r);
  m->setAttr("snabf.pass3_required", b.getUnitAttr());
  in.profile->setAttr("pass3_parameters", r);
  if (!security.empty()) {
    auto buf = llvm::MemoryBuffer::getFile(security);
    require(bool(buf), "cannot read security screen");
    auto text = buf.get()->getBuffer();
    checkSecurity(text, r);
    m->setAttr("snabf.security_screen", b.getStringAttr(text));
  }
  if (!audit.empty()) {
    auto json = jsonPass2(r);
    json["source_arithmetic_fraction"] = sourceFraction;
    json["requested_source_arithmetic_fraction"] = requestedFraction;
    json["requested_slots"] = int64_t(in.profile.getSlots());
    llvm::json::Array bits, scales, cuts;
    for (double x : s.bits)
      bits.push_back(x);
    for (double x : s.scales)
      scales.push_back(x);
    for (auto &c : s.cuts)
      cuts.push_back(c);
    json["continuous_q"] = std::move(bits);
    json["continuous_p"] = s.pBits;
    if (o.blockSearch()) {
      llvm::json::Array aux;
      for (double x : s.auxBits) aux.push_back(x);
      json["continuous_p_limbs"] = std::move(aux);
    }
    json["continuous_scales"] = std::move(scales);
    json["tightening_cuts"] = std::move(cuts);
    json["objective"] = o.costProfile ? real(r,"gadget_objective_us") : s.objective;
    if (o.costProfile) json["objective_units"] = "us";
    std::error_code ec;
    llvm::raw_fd_ostream out(audit, ec, llvm::sys::fs::OF_Text);
    require(!ec, "cannot write Pass 3 audit");
    out << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(json)));
  }
}
void verifyPass3(ModuleOp m) {
  bool profiled = false;
  bool ringPolicy = false;
  m.walk([&](ProfileOp f) {
    profiled |= f->hasAttr("pass3_parameters");
    ringPolicy |= f->hasAttr("snabf.pass4_policy");
  });
  // Also guards the ordinary HEIR lowering branch when the search attribute
  // has been stripped from a source-ring fallback.
  if (ringPolicy || m->hasAttr("snabf.pass4_required") ||
      m->hasAttr("snabf.pass4_effective_switching_key_security_bits") ||
      m->hasAttr("snabf.ring_search"))
    require(ringPolicy && m->getAttrOfType<UnitAttr>("snabf.pass4_required") &&
                m->getAttrOfType<IntegerAttr>(
                    "snabf.pass4_effective_switching_key_security_bits") &&
                m->getAttrOfType<DictionaryAttr>("snabf.ring_search"),
            "missing_ring_search_or_policy");
  if (!profiled && !m->hasAttr("snabf.pass3_required") &&
      !m->hasAttr("snabf.pass3") && !m->hasAttr("snabf.gadget_cost_profile"))
    return;
  Builder b(m.getContext());
  auto d = field<DictionaryAttr>(m, "snabf.pass3");
  auto in = extract(m);
  require(in.profile->getAttr("pass3_parameters") == d &&
              in.profile.getQAttr() == d.get("q") &&
              in.profile.getPAttr() == d.get("p") &&
              in.profile.getDefaultLogScaleAttr() == d.get("scale"),
          "Pass 3 needs HEIR specialization to the solved Q/P/scale");
  p::Solution s;
  s.q.assign(in.profile.getQ().begin(), in.profile.getQ().end());
  s.p.assign(in.profile.getP().begin(), in.profile.getP().end());
  s.scale = in.profile.getDefaultLogScale();
  for (auto e : field<ArrayAttr>(d, "edges"))
    s.k.push_back(num(cast<DictionaryAttr>(e), "K"));
  auto o = readOptions(field<DictionaryAttr>(d, "options"),m);
  auto v = p::replay(in.g, o, s);
  require(v.reason.empty(), v.reason);
  require(record(b, in, o, s) == d,
          "stale Pass 3 site, scale or error transfer");
  checkSecurity(field<StringAttr>(m, "snabf.security_screen").getValue(), d);
}
void bindPass3(DictionaryAttr d, llvm::json::Array &backend) {
  llvm::json::Array relins;
  for (auto r : field<ArrayAttr>(d, "edges")) {
    auto e = cast<DictionaryAttr>(r);
    if (field<StringAttr>(e, "op").getValue() == "relin")
      relins.push_back(jsonPass2(e));
  }
  for (auto &v : backend) {
    auto &obj = *v.getAsObject();
    auto pre = "lin" + std::to_string(*obj.getInteger("site")) + ".";
    for (auto role : {"baby", "giant"}) {
      llvm::json::Object map;
      for (auto r : field<ArrayAttr>(d, "edges")) {
        auto e = cast<DictionaryAttr>(r);
        auto id = field<StringAttr>(e, "ID").getValue();
        if (id.consume_front(pre + role))
          map[id.str()] = jsonPass2(e);
      }
      obj[std::string(role) == "baby" ? "Babies" : "Giants"] = std::move(map);
    }
  }
  (*backend.front().getAsObject())["Relins"] = std::move(relins);
}
} // namespace snabf::ir
