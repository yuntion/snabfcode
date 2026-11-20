#include "NoiseCertIR.h"
#include "Pass3IR.h"
#include "Pass4IR.h"
#include "Target/Lattigo/EmitGraph.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace snabf::ir {
namespace ng = noise;
namespace eg = lattigo_target::detail;
namespace {
int checkedInt(int64_t n) {
  require(n >= 0 && n <= std::numeric_limits<int>::max(), "out-of-range physical parameter");
  return int(n);
}

std::optional<ng::Exact> encoderScale(ModuleOp m, StringRef id, double log,
                                      const BoundEncodingScale &binding) {
  require(std::isfinite(log), "bad emitted plaintext scale exponent");
  std::optional<ng::Exact> exact;
  // DirectGo writes a round-trip binary64 literal; Bind calls Go math.Exp2.
  // Integer exponents in this range give EXACT binary64 powers of two.
  if (log == std::trunc(log) && log >= -1074 && log <= 1023)
    exact = pass2::exactBinary(std::ldexp(1., int(log)));
  if (binding) {
    if (auto s = binding(m, id, log)) {
      require(std::isfinite(*s) && *s > 0, "invalid materialized encoder scale");
      auto bound = pass2::exactBinary(*s);
      require(!exact || bound == *exact, "encoder binding disagrees with exact power of two");
      return bound;
    }
  }
  return exact;
}

std::vector<uint64_t> moduli(ArrayRef<int64_t> primes) {
  std::vector<uint64_t> out;
  for (auto q : primes) {
    require(q > 0, "negative materialized modulus");
    out.push_back(uint64_t(q));
  }
  return out;
}
} // namespace

ng::Topology checkedNoiseTopology(ModuleOp m, const BoundEncodingScale &binding) {
  const bool mixed = isMixed(m);
  if (mixed) checkRings(m);
  else { verifyPlan(m); verifyPass3(m); checkRingSearch(m); }
  m.walk([&](Operation *op) {
    if (auto b = op->getAttrOfType<DictionaryAttr>("snabf.ring_switch"))
      require(!field<BoolAttr>(b, "enabled").getValue(), "enabled bootstrap has no materialized B1 transfer");
  });
  auto par = field<DictionaryAttr>(m, mixed ? "snabf.mixed_parameters" : "snabf.pass3");
  const auto ctx = concreteSSAContext(m, par);
  // ringProgram attaches signed payloads and actual evaluation-key domains;
  // EmitGraph verifies them against the regenerated concrete plan. No
  // decoded_arithmetic_bound, profile noise table or empirical value is read.
  auto program = ringProgram(m);
  const eg::EmitGraph emitted(program);
  ng::Topology g;
  std::map<int64_t, size_t> domain;
  for (const auto &[n, d] : emitted.families) {
    domain.emplace(n, g.domains.size());
    g.domains.push_back({eg::str(d, "key_id").str(), checkedInt(n),
                         moduli(eg::ints(d, "q")), moduli(eg::ints(d, "p"))});
  }
  std::map<std::string, size_t> byID;
  std::vector<size_t> mapped;
  for (const auto &e : emitted.edges) {
    ng::Node n{};
    n.id = e.id;
    n.domain = domain.at(e.n);
    n.level = checkedInt(e.level);
    n.k = checkedInt(e.k);
    for (auto i : e.inputs) n.inputs.push_back(mapped.at(i));
    auto family = emitted.families.at(e.n);
    const auto &key = g.domains[n.domain].key_id;
    using O = eg::Op;
    switch (e.op) {
    case O::Source: {
      n.op = ng::Op::Source;
      n.input_draw = e.id; // EmitGraph has exactly one external input here
      auto exp = eg::num(family, "scale");
      require(exp > 0 && exp <= 1023, "invalid materialized default encoder scale");
      n.encoded_scale = pass2::exactBinary(std::ldexp(1., int(exp)));
      break;
    }
    case O::Drop: n.op = ng::Op::Drop; break;
    case O::Mul: n.op = ng::Op::CtMul; break;
    case O::Mask: case O::Coefficient: case O::AddPlain: {
      // These operands alone are materialized by epoch.Bind. Match the Go
      // emitter's subtraction of two *binary64* exponents before Exp2.
      const auto parentLog = emitted.edges.at(e.inputs[0]).scale;
      const auto plainLog = e.op == O::AddPlain ? parentLog : e.scale - parentLog;
      n.encoded_scale = encoderScale(m, e.id, plainLog, binding);
      if (e.op != O::AddPlain && e.values.size() == 1) {
        n.op = ng::Op::ScalarMulPlain;
        n.real_scalar = e.values.front();
      } else {
        n.op = e.op == O::AddPlain ? ng::Op::AddPlain : ng::Op::VectorMulPlain;
        n.plain_values.assign(e.values.begin(), e.values.end());
      }
      break;
    }
    case O::Add: {
      n.op = ng::Op::Add;
      const auto args = n.inputs;
      // Actual DirectGo is a left fold. Every binary addition needs its
      // own residual; the final one owns the original SSA edge ID.
      size_t acc = args.front();
      for (size_t j = 1; j + 1 < args.size(); ++j) {
        auto fold = n;
        fold.id += "#add" + std::to_string(j);
        fold.inputs = {acc, args[j]};
        acc = g.nodes.size();
        g.nodes.push_back(std::move(fold));
      }
      n.inputs = {acc, args.back()};
      break;
    }
    case O::Rescale: n.op = ng::Op::Rescale; break;
    case O::Relin: n.op = ng::Op::Relin; n.evk = key + ":relin"; break;
    case O::Baby: case O::Giant:
      n.op = ng::Op::Rotate;
      n.rotation = e.shift;
      n.evk = key + ":rotation:" + std::to_string(e.shift);
      break;
    case O::Down:
      n.op = ng::Op::Down;
      n.evk = eg::str(emitted.switchingDomain, "source_key").str() + ":down_to:" + key;
      break;
    default: throw ContractError("opcode has no noise topology adapter: " + e.id);
    }
    mapped.push_back(g.nodes.size());
    require(byID.emplace(e.id, g.nodes.size()).second, "duplicate bound edge ID");
    g.nodes.push_back(std::move(n));
  }
  m.walk([&](SinkOp op) {
    auto bound = ctx.values.lookup(op.getInput());
    require(bool(bound), "sink missing from independently extracted SSA map");
    const auto ix = byID.at(eg::str(bound, "ID").str());
    require(mapped.at(emitted.sink) == ix && int64_t(op.getInput().getType().getN()) ==
            g.domains[g.nodes[ix].domain].ring_n, "emitted sink does not match SSA output");
    g.outputs.push_back({"sink" + std::to_string(g.outputs.size()), ix});
  });
  require(!g.outputs.empty(), "empty returned SSA sink inventory");
  return g;
}

namespace {
namespace j = llvm::json;
j::Value number(std::optional<double> n) {
  return n && std::isfinite(*n) ? j::Value(*n) : j::Value(nullptr);
}
j::Value exactText(const std::string &s) { return s.empty() ? j::Value(nullptr) : j::Value(s); }
const char *sourceKind(pass2::NoiseSourceKind k) {
  switch (k) {
  case pass2::NoiseSourceKind::FreshEncrypt: return "fresh_encrypt";
  case pass2::NoiseSourceKind::FreshEncode: return "fresh_encode";
  case pass2::NoiseSourceKind::ProvenIndependent: return "proven_independent";
  }
  throw ContractError("invalid audited source kind");
}
const char *depKind(pass2::DependentKind k) {
  switch (k) {
  case pass2::DependentKind::RescaleRound: return "rescale_round";
  case pass2::DependentKind::GadgetTrunc: return "gadget_trunc";
  case pass2::DependentKind::SharedEvkSwitch: return "shared_evk_switch";
  case pass2::DependentKind::DownConvRound: return "downconv_round";
  case pass2::DependentKind::BootstrapNoise: return "bootstrap_noise";
  case pass2::DependentKind::UnknownDependent: return "unknown_dependent";
  }
  throw ContractError("invalid audited dependent kind");
}

// Stage the audit beside the destination; never leave a previous positive
// report partially overwritten or report a rejected certificate as issued.
void publishAudit(StringRef path, j::Object data) {
  require(!path.empty() && path != "-", "noise audit requires an output file");
  const std::filesystem::path out(path.str());
  if (out.has_parent_path()) std::filesystem::create_directories(out.parent_path());
  auto st = std::filesystem::symlink_status(out).type();
  require(st == std::filesystem::file_type::regular || st == std::filesystem::file_type::not_found,
          "unsafe noise audit destination");
  auto tmp = llvm::sys::fs::TempFile::create(path + ".%%%%%%.tmp");
  if (!tmp) throw ContractError("cannot stage noise audit: " + llvm::toString(tmp.takeError()));
  bool failed;
  {
    llvm::raw_fd_ostream stream(tmp->FD, false);
    stream << llvm::formatv("{0:2}\n", j::Value(std::move(data)));
    stream.flush();
    failed = stream.has_error();
    if (failed) stream.clear_error();
  }
  if (failed) {
    llvm::consumeError(tmp->discard());
    throw ContractError("cannot write noise audit");
  }
  if (auto err = tmp->keep(path)) throw ContractError("cannot publish noise audit: " + llvm::toString(std::move(err)));
}
} // namespace

j::Object noiseAuditJSON(const ng::CertAuditReport &r) {
  j::Object out;
  out["schema"] = "snabf.noise_cert.v1";
  out["mode"] = r.mode == ng::Mode::Engineering ? "engineering" : "probability";
  out["all_contracts_present"] = r.all_contracts_present;
  out["event_budget_passed"] = r.event_budget_passed;
  out["all_sinks_margin_passed"] = r.all_sinks_margin_passed;
  out["probabilistic_cert_issued"] = r.probabilistic_cert_issued;
  out["total_delta_valid"] = number(r.total_delta_valid);
  out["sum_delta_sinks"] = number(r.sum_delta_sinks);
  out["target_failure_prob"] = number(r.target_failure_prob);
  out["common_event"] = exactText(r.common_event);
  out["common_event_proof"] = exactText(r.common_proof);
  out["joint_independence_proof"] = exactText(r.joint_proof);
  j::Array ports, linear, deps, events, missing, errors, sites, edges, domains;
  for (const auto &d : r.domains) {
    j::Array q, p;
    // Strings retain the integer exactly in JSON clients using binary64.
    for (auto x : d.q) q.push_back(std::to_string(x));
    for (auto x : d.p) p.push_back(std::to_string(x));
    domains.emplace_back(j::Object{{"key_id", d.key_id}, {"ring_n", d.ring_n},
                                   {"q", std::move(q)}, {"p", std::move(p)}});
  }
  for (const auto &e : r.edge_inventory)
    edges.emplace_back(j::Object{{"node_id", e.id}, {"key_id", e.key_id}, {"evk", e.evk},
        {"ring_n", e.ring_n}, {"level", e.level}, {"k", e.k}, {"physical_scale_exact", exactText(e.physical_scale_exact)}});
  for (const auto &s : r.sinks)
    ports.emplace_back(j::Object{{"sink_id", s.sink_id}, {"ring_n", s.ring_n},
        {"physical_scale_exact", exactText(s.physical_scale_exact)},
        {"actual_physical_scale", number(s.actual_physical_scale)},
        {"deterministic_bias_b", number(s.deterministic_bias_b)},
        {"total_variance_proxy_v", number(s.total_variance_proxy_v)},
        {"allocated_delta_sink", number(s.allocated_delta_sink)},
        {"required_tolerance_units", number(s.required_tolerance_units)},
        {"certified_bound_units", number(s.certified_bound_units)}, {"margin_passed", s.margin_passed}});
  for (const auto &[sink, f] : r.sink_forms) {
    for (const auto &[src, t] : f.linearTerms())
      linear.emplace_back(j::Object{{"sink_id", sink}, {"source_id", src}, {"kind", sourceKind(t.kind)},
          {"signed_coeff", t.signed_coeff.get_str()}, {"base_var_proxy", t.base_var_proxy.get_str()},
          {"formula_ref", t.formula_ref}, {"joint_proof", exactText(t.joint_proof)}});
    for (const auto &[site, t] : f.dependentTerms())
      deps.emplace_back(j::Object{{"sink_id", sink}, {"site_id", site}, {"kind", depKind(t.kind)},
          {"shared_key_id", t.shared_key_id}, {"sigma_proxy", t.sigma_proxy.get_str()}, {"formula_ref", t.formula_ref}});
  }
  for (const auto &[key, p] : r.validity_events)
    events.emplace_back(j::Object{{"site_id", key.first}, {"reason", key.second}, {"failure_prob_exact", p.get_str()}});
  for (const auto &s : r.sites)
    sites.emplace_back(j::Object{{"site_id", s.id}, {"primitive", ng::name(s.role)}, {"formula_ref", exactText(s.formula_ref)}});
  for (const auto &s : r.missing_contract_terms) missing.push_back(s);
  for (const auto &s : r.errors) errors.push_back(s);
  out["domains"] = std::move(domains);
  out["edges"] = std::move(edges);
  out["sites"] = std::move(sites);
  out["sinks"] = std::move(ports);
  out["merged_same_source_terms"] = std::move(linear);
  out["conservative_holder_terms"] = std::move(deps);
  out["validity_events"] = std::move(events);
  out["missing_contract_terms"] = std::move(missing);
  out["errors"] = std::move(errors);
  return out;
}

void auditNoisePlan(ModuleOp m, StringRef path, bool certify, unsigned bits, double target) {
  ng::CertAuditReport r;
  r.mode = certify ? ng::Mode::Probability : ng::Mode::Engineering;
  r.target_failure_prob = target;
  try {
    require(bits <= 16384 && std::isfinite(target) && target > 0 && target <= ng::theoremBudget,
            "invalid noise precision or failure budget");
    if (!m->hasAttr("snabf.pass3")) {
      // Legacy pre-specialization check: never invent a lowering, physical
      // scales or an empty set of primitive obligations from a high-level IR.
      verifyPlan(m); verifyPass3(m); checkRingSearch(m);
      r.missing_contract_terms.insert("common/validated_joint_law_and_domain");
      r.missing_contract_terms.insert("common/materialized_primitive_topology_and_encoder_scales");
      size_t index = 0;
      m.walk([&](Operation *op) {
        if (auto sink = dyn_cast<SinkOp>(op)) {
          auto &s = r.sinks.emplace_back();
          s.sink_id = "sink" + std::to_string(r.sinks.size() - 1);
          s.ring_n = checkedInt(sink.getInput().getType().getN());
          r.missing_contract_terms.insert("sink/" + s.sink_id + "/reference_approx_and_decode");
        } else if (llvm::any_of(op->getResultTypes(), [](Type t) { return isa<CtType>(t); })) {
          r.missing_contract_terms.insert("unexpanded/" + std::to_string(index++) + "/" + name(op).str());
        }
      });
      require(index && !r.sinks.empty(), "empty legacy ciphertext/sink inventory");
    } else {
      auto graph = checkedNoiseTopology(m);
      ng::Options opts{r.mode, int(bits), target, {}};
      // Reserve at least half of the budget for the common validity event.
      double perSink = pass2::lowerDouble(pass2::exactBinary(target) /
          (ng::Exact(2) * ng::Exact(std::to_string(graph.outputs.size()))));
      for (const auto &s : graph.outputs) opts.sink_delta.emplace(s.id, perSink);
      r = ng::verifyNoiseCertificate(graph, opts); // no profile-to-proof path
    }
  } catch (const std::exception &e) { r.errors.emplace_back(e.what()); }
  auto data = noiseAuditJSON(r);
  data["precision_bits"] = int64_t(bits);
  publishAudit(path, std::move(data));
  require(r.errors.empty(), r.errors.empty() ? "" : "invalid noise replay: " + r.errors.front());
  require(r.accepts(), "noise certificate denied; see " + path);
}

void auditNoisePlan(ModuleOp m, StringRef path, bool certify, unsigned bits) {
  auditNoisePlan(m, path, certify, bits, ng::theoremBudget);
}
} // namespace snabf::ir
