#include "Pipeline.h"
#include "Pass2IR.h"
#include "AffineIntervalAnalyzer.h"
#include "pass1/NativeRouting.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <set>

namespace snabf::ir {
double upperProduct(double a,double b) {
  APFloat x(a); x.multiply(APFloat(b),APFloat::rmTowardPositive);
  require(x.isFinite(),"interval product overflow");return x.convertToDouble();
}
double upperSum(double a,double b) {
  APFloat x(a); x.add(APFloat(b),APFloat::rmTowardPositive);
  require(x.isFinite(),"interval sum overflow");return x.convertToDouble();
}
namespace {
bool affineMode(ModuleOp m) {
  if (!m->hasAttr("snabf.interval_mode")) return false;
  auto mode=field<StringAttr>(m,"snabf.interval_mode").getValue();
  require(mode=="naive" || mode=="affine", "invalid snabf.interval_mode");
  if (mode=="naive") return false;
  require(m->hasAttr("snabf.activation_sources"), "affine interval mode requires a source-proven logical graph");
  return true;
}

struct AffineFact { affine::Interval domain; DictionaryAttr report; };
using AffineFacts = llvm::DenseMap<Value,AffineFact>;

// This is the encoder's single periodic logical vector map, not merely a
// divisibility test. PackedMulti, padding, and unknown maps cannot alias SSA
// symbols here. Each diagonal replica is checked separately by the core.
int64_t periodicWidth(CtType t, ProfileOp profile) {
  auto logical=t.getLogical();
  require(profile && logical.hasStaticShape() && logical.getRank()==1,
          "affine interval: expected a static rank-one periodic layout");
  auto width=logical.getNumElements(), period=int64_t(profile.getSlots());
  require(width>0 && period>0 && period%width==0 && (t.getN()/2)%period==0,
          "affine interval: invalid physical/logical period");
  auto expected = "#tensor_ext.original_type<originalType = " + spelling(logical) +
      ", layout = #tensor_ext.layout<\"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod " +
      std::to_string(width) + " = 0 and 0 <= i0 <= " + std::to_string(width-1) +
      " and 0 <= slot <= " + std::to_string(period-1) + " }\">>";
  require(spelling(profile.getLayout())==expected, "affine interval: unproven periodic source layout");
  return width;
}

DictionaryAttr affineReport(Builder& b, const affine::AffineIntervalAnalyzer& engine,
                            affine::AffineIntervalAnalyzer::Value v, StringRef rule, double error) {
  SmallVector<double> lo, hi, naiveLo, naiveHi, rem, rounding;
  SmallVector<int64_t> symbols, condensed;
  for (const auto& s:engine.slots(v)) {
    lo.push_back(s.enclosure.lo); hi.push_back(s.enclosure.hi);
    naiveLo.push_back(s.naive.lo); naiveHi.push_back(s.naive.hi);
    rem.push_back(s.form.remainder); rounding.push_back(s.roundingRadius);
    symbols.push_back(s.form.coefficients.size()); condensed.push_back(s.condensed);
  }
  NamedAttrList d;
  d.set("schema",b.getStringAttr("affine-logical-slots.v1"));
  d.set("scope",b.getStringAttr("ideal_and_admitted_real_polynomials_before_ckks"));
  d.set("rule",b.getStringAttr(rule)); d.set("local_polynomial_error_cap",b.getF64FloatAttr(error));
  d.set("precision_bits",b.getI64IntegerAttr(engine.precision()));
  d.set("symbol_limit",b.getI64IntegerAttr(engine.symbolLimit()));
  d.set("lo",b.getDenseF64ArrayAttr(lo)); d.set("hi",b.getDenseF64ArrayAttr(hi));
  d.set("naive_lo",b.getDenseF64ArrayAttr(naiveLo)); d.set("naive_hi",b.getDenseF64ArrayAttr(naiveHi));
  d.set("symbols",b.getDenseI64ArrayAttr(symbols)); d.set("condensed",b.getDenseI64ArrayAttr(condensed));
  d.set("remainder",b.getDenseF64ArrayAttr(rem)); d.set("rounding_radius",b.getDenseF64ArrayAttr(rounding));
  return d.getDictionary(b.getContext());
}

// Topological, choice-independent pass. By induction a candidate family is
// covered if EVERY admitted real polynomial satisfies its own same-input
// residual cap on the incoming enclosure. These are not CKKS noise bounds.
AffineFacts affineFacts(ModuleOp m, bool reports) {
  AffineFacts facts;
  if (!affineMode(m)) return facts;
  Builder b(m.getContext());
  affine::AffineIntervalAnalyzer engine;
  using Handle=affine::AffineIntervalAnalyzer::Value;
  llvm::DenseMap<Value,Handle> edges;
  llvm::DenseMap<Value,std::pair<double,Handle>> ports;
  auto input=[&](Value v) {
    auto it=edges.find(v); require(it!=edges.end(),"non-topological affine SSA input");
    return it->second;
  };
  m.walk([&](Operation* o) {
    if (o->getNumResults()!=1 || !isa<CtType>(o->getResult(0).getType())) return;
    Value v=o->getResult(0); auto t=cast<CtType>(v.getType());
    auto profile=SymbolTable::lookupNearestSymbolFrom<ProfileOp>(o,t.getProfile());
    int64_t width=periodicWidth(t,profile);
    StringRef rule; double error=0;
    auto eval=[&]() -> Handle {
      if (auto s=dyn_cast<SourceOp>(o)) {
        rule="source_periodic";
        double bound=field<FloatAttr>(o,"input_bound").getValueAsDouble();
        require(std::isfinite(bound) && bound>0,"invalid affine source bound");
        auto ext=s.getCipher().getDefiningOp();
        require(ext && name(ext)=="tensor.extract" && ext->getNumOperands()==2,"unbound affine source port");
        auto arg=dyn_cast<BlockArgument>(ext->getOperand(0));
        auto func=o->getParentOfType<func::FuncOp>();
        auto idx=ext->getOperand(1).getDefiningOp();
        require(arg && func && arg.getOwner()==&func.getBody().front() && idx && name(idx)=="arith.constant" &&
                field<IntegerAttr>(idx,"value").getInt()==0 &&
                func.getArgAttr(arg.getArgNumber(),"tensor_ext.original_type")==profile.getLayout(),
                "affine interval: source is not the witnessed ct=0 input port");
        if (auto it=ports.find(arg); it!=ports.end()) {
          require(it->second.first==bound && engine.slots(it->second.second).size()==size_t(width),
                  "conflicting affine domains for one input port");
          return it->second.second;
        }
        auto id=engine.source(std::vector<affine::Interval>(width,{-bound,bound}));
        ports.try_emplace(arg,std::make_pair(bound,id)); return id;
      }
      if (auto l=dyn_cast<LintransOp>(o)) {
        rule="signed_periodic_diagonals";
        auto offsets=l.getDiagonals(); auto w=preparedWeights(l.getTable(),offsets);
        auto shape=w.getType().getShape(); auto e=w.getElementType();
        require(shape.size()==2 && shape[0]==int64_t(offsets.size()) && shape[1]==int64_t(profile.getSlots()) &&
                (e.isF32() || e.isF64()),"affine interval: unbound binary32/64 diagonal table");
        std::vector<double> weights; weights.reserve(w.getNumElements());
        for (auto a:w.getValues<APFloat>()) weights.push_back(a.convertToDouble());
        return engine.periodicDiagonals(input(l.getInput()),shape[1],{offsets.data(),offsets.size()},weights);
      }
      if (auto p=dyn_cast<PolyEvalOp>(o)) {
        auto kind=field<StringAttr>(p,"activation").getValue();
        require(kind=="silu" || kind=="gelu","unsupported affine logical activation");
        error=field<FloatAttr>(p,"approx_budget").getValueAsDouble();
        rule=kind=="silu"?"silu_halfline_residual":"erf_gelu_halfline_residual";
        return engine.activate(input(p.getInput()),kind=="silu"?affine::Activation::SiLU:affine::Activation::ErfGeLU,error);
      }
      if (auto s=dyn_cast<StepOp>(o); s && s.getKind()=="ckks.add" && o->getNumOperands()==2) {
        rule="residual_add"; return engine.add(input(o->getOperand(0)),input(o->getOperand(1)));
      }
      if (auto c=dyn_cast<ConvertOp>(o)) {
        require(c.getLayoutContract()=="trace-invariant","unsupported affine layout conversion");
        // A conversion may preserve this logical vector only under the existing
        // trace-invariant single-lane convention; never infer packed-lane sharing.
        for (auto desc:{c.getProvidedAttr(),c.getRequiredAttr()}) {
          require(bool(desc),"missing affine conversion layout proof"); auto d=desc.getFields();
          require(field<StringAttr>(d,"packing_mode").getValue()=="periodic_single" &&
                  field<IntegerAttr>(d,"logical_width").getInt()==width && field<IntegerAttr>(d,"stride").getInt()==1 &&
                  field<IntegerAttr>(d,"period").getInt()==int64_t(profile.getSlots()) &&
                  field<DenseI64ArrayAttr>(d,"offsets").asArrayRef()==ArrayRef<int64_t>({0}) &&
                  field<DenseI64ArrayAttr>(d,"active_lanes").asArrayRef()==ArrayRef<int64_t>({0}) &&
                  field<DenseI64ArrayAttr>(d,"mask_signature").empty(),"unsupported affine conversion lane map");
        }
        rule="trace_periodic_identity"; return input(c.getInput());
      }
      throw ContractError("unsupported affine SSA transfer: "+name(o).str());
    };
    Handle id=eval(); require(engine.slots(id).size()==size_t(width),"affine logical slot count changed");
    edges.try_emplace(v,id);
    facts.try_emplace(v,AffineFact{engine.domain(id),reports?affineReport(b,engine,id,rule,error):DictionaryAttr{}});
  });
  return facts;
}

DictionaryAttr inventory(Builder &b, const RoutingPlan &p) {
  SmallVector<Attribute> keys;
  for (auto &[k, refs] : p.referenceCounts)
    keys.push_back(b.getDictionaryAttr(
        {b.getNamedAttr("scope", b.getStringAttr(k.first)),
         b.getNamedAttr("galois", b.getI64IntegerAttr(k.second)),
         b.getNamedAttr("refs", b.getI64IntegerAttr(refs))}));
  return b.getDictionaryAttr(
      {b.getNamedAttr("keys", b.getArrayAttr(keys)),
       b.getNamedAttr("bytes", b.getI64IntegerAttr(p.keyBytes)),
       b.getNamedAttr("objective", b.getF64FloatAttr(p.objective))});
}

struct Flow {
  double bound = 0;
  double lo = 0, hi = 0, approx = 0;
  bool pass2 = false;
  DictionaryAttr conditional;
  DictionaryAttr affineSlots;
  int64_t level = 0, scale = 0, depth = 0;
  SmallVector<int64_t> witness;
  bool operator==(const Flow &) const = default;
};
struct State : AnalysisState {
  using AnalysisState::AnalysisState;
  std::optional<Flow> value;
  void print(llvm::raw_ostream &os) const override {
    if (value)
      os << value->depth;
    else
      os << "uninitialized";
  }
};
// Outward conversion, preserving exact dyadic values when representable.
double up(long double value) {
  double d = static_cast<double>(value);
  if (d < value) d = std::nextafter(d, INFINITY);
  require(std::isfinite(d), "interval overflow");
  return d;
}
double norm(DenseFPElementsAttr weights, bool matrix) {
  auto cols =
      matrix ? weights.getType().getShape()[1] : weights.getNumElements();
  std::vector<double> sums(cols);
  size_t i = 0;
  for (auto f : weights.getValues<APFloat>()) {
    auto &s = sums[i++ % cols];
    s = upperSum(s,std::abs(f.convertToDouble()));
  }
  return *std::max_element(sums.begin(), sums.end());
}
struct BudgetFlow : DataFlowAnalysis {
  using DataFlowAnalysis::DataFlowAnalysis;
  AffineFacts facts;
  LogicalResult initialize(Operation *top) override {
    facts=affineFacts(cast<ModuleOp>(top),true);
    top->walk([&](Operation *o) {
      if (o->getNumResults() == 1 && isa<CtType>(o->getResult(0).getType()))
        (void)visit(getProgramPointBefore(o));
    });
    return success();
  }
  LogicalResult visit(ProgramPoint *point) override {
    auto o = point->getNextOp();
    auto t = cast<CtType>(o->getResult(0).getType());
    auto profile =
        SymbolTable::lookupNearestSymbolFrom<ProfileOp>(o, t.getProfile());
    auto native = decodeCT(field<TypeAttr>(o, "native").getValue());
    auto module = o->getParentOfType<ModuleOp>();
    const bool logical = module->hasAttr("snabf.activation_sources");
    bool epoch=profile && profile->hasAttr("snabf.target_epoch");
    require(profile && (epoch ? native.n>t.getN() && native.n%t.getN()==0 : native.n==t.getN()) && (logical || native.level == t.getLevel()) &&
                (epoch ? ArrayRef(native.chain).take_front(std::min(native.chain.size(),profile.getQ().size()))==profile.getQ() : ArrayRef(native.chain)==profile.getQ()),
            "stale native ciphertext/profile");
    SmallVector<Flow, 2> inputs;
    for (auto v : o->getOperands())
      if (isa<CtType>(v.getType())) {
        auto state = getOrCreateFor<State>(point, v);
        if (!state->value) return success();
        inputs.push_back(*state->value);
      }
    Flow f;
    if (!inputs.empty()) f = inputs[0];
    f.pass2 = logical;
    f.conditional={}; // No implicit transfer through a primitive with no contract.
    if (auto src = dyn_cast<SourceOp>(o)) {
      require(
          field<TypeAttr>(o, "native").getValue() == src.getCipher().getType(),
          "changed source ABI type");
      auto ext = src.getCipher().getDefiningOp();
      require(ext && name(ext) == "tensor.extract",
              "missing packed input extraction");
      auto arg = dyn_cast<BlockArgument>(ext->getOperand(0));
      auto func = src->getParentOfType<func::FuncOp>();
      require(
          arg && func &&
              func.getArgAttr(arg.getArgNumber(), "tensor_ext.original_type") ==
                  profile.getLayout(),
          "profile layout disagrees with HEIR input");
      auto m = src->getParentOfType<ModuleOp>();
      auto pars = decodeParameters(field<Attribute>(m, "ckks.schemeParam"));
      auto iface = field<DictionaryAttr>(func, "heir.interface");
      require(
          profile.getKeyId() == field<StringAttr>(iface, "func_name").str() +
                                    "__configure:" + native.key &&
              int64_t(profile.getSlots()) ==
                  field<IntegerAttr>(m, "scheme.requested_slot_count")
                      .getInt() &&
              ArrayRef(pars.p) == profile.getP() &&
              int64_t(profile.getDefaultLogScale()) == pars.logScale,
          "stale key family, P chain, scale or packing");
      f.bound = field<FloatAttr>(src, "input_bound").getValueAsDouble();
      require(std::isfinite(f.bound) && f.bound > 0, "invalid input bound");
      f.level = t.getLevel();
      f.scale = native.logScale;
    } else if (auto c = dyn_cast<ConvertOp>(o)) {
      require(epoch && c.getLayoutContract()=="trace-invariant" && f.level>=t.getLevel(),"unsupported conversion transfer");
      f.level=t.getLevel(); // Pass 3 materializes a use-local level drop, with unchanged scale.
    } else if (auto lt = dyn_cast<LintransOp>(o)) {
      auto weights = preparedWeights(lt.getTable(), lt.getDiagonals());
      auto opNorm = norm(weights, true);
      f.bound = upperProduct(f.bound,opNorm);
      f.approx = upperProduct(f.approx,opNorm);
      --f.level;
      auto choice = field<DictionaryAttr>(o, "selected");
      f.depth += field<IntegerAttr>(choice, "rotation_depth").getInt();
      f.witness.push_back(field<IntegerAttr>(o, "site").getInt());
    } else if (auto poly = dyn_cast<PolyEvalOp>(o)) {
      if (logical) {
        auto c = field<DictionaryAttr>(o,"poly_selected");
        auto tr=field<DictionaryAttr>(c,"transfer");
        f.conditional=field<IntegerAttr>(tr,"input_level").getInt()==f.level ? tr : transferRecord(poly,c,f.level);
        auto domain = field<DenseF64ArrayAttr>(c,"input_interval");
        require(domain[0]<=f.lo && f.hi<=domain[1],"activation domain does not cover incoming trajectory");
        auto out=field<DenseF64ArrayAttr>(c,"output_interval");
        f.lo=out[0]; f.hi=out[1]; f.bound=std::max(-f.lo,f.hi);
        f.approx=upperSum(upperProduct(field<FloatAttr>(c,"gain").getValueAsDouble(),f.approx),field<FloatAttr>(c,"activation_error").getValueAsDouble());
        f.level-=field<IntegerAttr>(c,"delta_level").getInt(); f.scale=profile.getDefaultLogScale();
      } else {
      auto cs = poly.getCoefficients();
      require(cs.size() > 1, "expected a nonconstant polynomial");
      double x = f.bound, prev = 1, cur = x, sum = 0;
      for (auto [j, a] : llvm::enumerate(cs)) {
        double term = j == 0 ? 1 : (x <= 1 ? 1 : cur);
        sum = up((long double)sum +
                 std::abs(cast<FloatAttr>(a).getValueAsDouble()) *
                     (long double)term);
        if (j > 0 && x > 1) {
          auto next = up(2 * (long double)x * cur + prev);
          prev = cur;
          cur = next;
        }
      }
      f.bound = sum;
      f.level -= std::bit_width(cs.size() - 1);
      f.scale = profile.getDefaultLogScale();
      }
    } else if (auto step = dyn_cast<StepOp>(o)) {
      auto k = step.getKind();
      auto attrs = field<DictionaryAttr>(o, "native_props");
      if (k == "ckks.mul_plain") {
        require(o->getNumOperands() == 2 && inputs.size() == 1,
                "bad mul_plain signature");
        auto v = o->getOperand(1);
        f.bound = up((long double)f.bound * norm(plainWeights(v), false));
        // The HEIR plaintext type carries its nominal encoding scale.
        auto text = spelling(v.getType());
        auto pos = text.find("scaling_factor = ");
        require(pos != std::string::npos, "missing mask scale");
        StringRef suffix = StringRef(text).drop_front(pos + 17);
        int64_t scale;
        require(!suffix.consumeInteger(10, scale), "bad mask scale");
        f.scale += scale;
      } else if (k == "ckks.add" && logical) {
        require(inputs.size()==2 && inputs[0].scale==inputs[1].scale,"incompatible residual add");
        f.level=std::min(f.level,inputs[1].level);
        f.bound=upperSum(f.bound,inputs[1].bound);
        f.approx=upperSum(f.approx,inputs[1].approx);
        if(inputs[1].depth>f.depth){f.depth=inputs[1].depth;f.witness=inputs[1].witness;}
      } else if (k == "ckks.mul") {
        require(inputs.size() == 2 && inputs[0].level == inputs[1].level,
                "unaligned multiplication");
        f.bound = up((long double)f.bound * inputs[1].bound);
        f.scale += inputs[1].scale;
        if (inputs[1].depth > f.depth) {
          f.depth = inputs[1].depth;
          f.witness = inputs[1].witness;
        }
      } else if (k == "ckks.rescale") {
        --f.level;
        f.scale -= profile.getDefaultLogScale();
      } else if (k == "ckks.level_reduce")
        f.level -= field<IntegerAttr>(attrs, "levelToDrop").getInt();
      else
        require(k == "ckks.relinearize", "unmodeled CKKS transfer: " + k);
    } else
      throw ContractError("no validated transfer for " + name(o).str());
    if (!logical || !isa<PolyEvalOp>(o)) { f.lo=-f.bound; f.hi=f.bound; }
    if (auto it=facts.find(o->getResult(0)); it!=facts.end()) {
      // The old transfer follows the selected real polynomial. Independently,
      // the per-slot proof covers every admissible choice including this one.
      f.lo=std::max(f.lo,it->second.domain.lo); f.hi=std::min(f.hi,it->second.domain.hi);
      require(f.lo<=f.hi,"empty selected/affine trajectory intersection");
      f.bound=std::max(-f.lo,f.hi); f.affineSlots=it->second.report;
    }
    require(f.level==t.getLevel() && (logical ? f.scale==int64_t(profile.getDefaultLogScale()) : f.scale==native.logScale),
            "recomputed level/nominal scale disagree at " + name(o));
    auto target = getOrCreate<State>(o->getResult(0));
    if (target->value != f) {
      target->value = f;
      propagateIfChanged(target, ChangeResult::Change);
    }
    return success();
  }
};
BudgetAttr budget(Builder &b, const Flow &f) {
  NamedAttrList d;
  d.set("level", b.getI64IntegerAttr(f.level));
  d.set("nominal_log_scale", b.getI64IntegerAttr(f.scale));
  d.set("interval", b.getDenseF64ArrayAttr({f.lo,f.hi}));
  if (f.affineSlots) d.set("affine_slots",f.affineSlots);
  if (f.pass2) {
    d.set("approx_error",b.getF64FloatAttr(f.approx));
    d.set("canonical_approx_component",b.getF64FloatAttr(std::ldexp(f.approx,f.scale)));
  }
  d.set("rotation_path", b.getI64IntegerAttr(f.depth));
  for (auto k : {"canonical_bias", "variance_proxy", "decoded_error",
                 "validity_failure"})
    d.set(k, b.getUnitAttr());
  if(f.conditional){
    for(auto k:{"canonical_bias","variance_proxy","validity_failure"})d.set(k,f.conditional.get(k));
    d.set("arithmetic_contract",f.conditional.get("status"));
  }
  return BudgetAttr::get(b.getContext(), d.getDictionary(b.getContext()));
}
std::vector<RouteDepthCut> analyze(ModuleOp m, bool checkOnly) {
  DataFlowSolver solver;
  solver.load<BudgetFlow>();
  require(succeeded(solver.initializeAndRun(m)), "budget dataflow failed");
  Builder b(m.getContext());
  std::vector<RouteDepthCut> cuts;
  unsigned sinks = 0;
  m.walk([&](Operation *o) {
    bool sink = isa<SinkOp>(o);
    if (!sink &&
        (o->getNumResults() != 1 || !isa<CtType>(o->getResult(0).getType())))
      return;
    auto v = sink ? o->getOperand(0) : o->getResult(0);
    auto s = solver.lookupState<State>(v);
    require(s && s->value, "unresolved budget at " + name(o));
    auto attr = budget(b, *s->value);
    if (checkOnly)
      require(attr == o->getAttr("budget"), "stale budget at " + name(o));
    else
      o->setAttr("budget", attr);
    if (isa<LintransOp>(o) && s->value->pass2) {
      auto in=solver.lookupState<State>(o->getOperand(0)); require(in && in->value,"missing linear operand budget");
      auto ib=budget(b,*in->value);
      if(checkOnly) require(o->getAttr("input_budget")==ib,"stale downstream linear input budget");
      else o->setAttr("input_budget",ib);
    }
    if (sink) {
      ++sinks;
      if (auto cap = o->getAttrOfType<IntegerAttr>("rotation_path_limit")) {
        auto limit = cap.getInt();
        require(limit >= 0 && limit <= INT32_MAX,
                "invalid sink rotation limit");
        if (s->value->depth > limit)
          cuts.push_back({{s->value->witness.begin(), s->value->witness.end()},
                          unsigned(limit)});
      }
    }
  });
  require(sinks > 0, "missing snabf sink");
  return cuts;
}
}  // namespace
double linearNorm(DenseFPElementsAttr w,bool m){return norm(w,m);}
llvm::DenseMap<Value,std::pair<double,double>> affinePlanningDomains(ModuleOp m) {
  llvm::DenseMap<Value,std::pair<double,double>> domains;
  for (const auto& [v,f]:affineFacts(m,false))
    domains.try_emplace(v,std::make_pair(f.domain.lo,f.domain.hi));
  return domains;
}

static bool hasLinear(ModuleOp m) {
  bool found = false;
  m.walk([&](LintransOp) { found = true; });
  return found;
}
// A round stores indices in the published catalog, not HiGHS column IDs.
// The checker can replay every SSA violation without running HiGHS.
static DictionaryAttr roundRecord(Builder& b, const std::vector<size_t>& ids,
                                  const std::vector<RouteDepthCut>& violations) {
  SmallVector<int64_t> choices(ids.begin(),ids.end()); SmallVector<Attribute> cuts;
  for(const auto& v:violations) {
    SmallVector<int64_t> sites(v.sites.begin(),v.sites.end());
    cuts.push_back(b.getDictionaryAttr({b.getNamedAttr("sites",b.getDenseI64ArrayAttr(sites)),
                                        b.getNamedAttr("limit",b.getI64IntegerAttr(v.limit))}));
  }
  return b.getDictionaryAttr({b.getNamedAttr("choices",b.getDenseI64ArrayAttr(choices)),
    b.getNamedAttr("violations",b.getArrayAttr(cuts)),
    b.getNamedAttr("reason",b.getStringAttr(cuts.empty()?"feasible":"ssa_rotation_path"))});
}
void publishPass1(ModuleOp m) {
  auto boot = evaluateBootstraps(m, false);
  Builder b(m.getContext());
  if (hasLinear(m))
    publishRoutes(Catalog(m), b);
  else
    require(boot, "no linear or bootstrap candidate sites");
}
static void selectRoutes(ModuleOp m, bool independent) {
  auto boot = evaluateBootstraps(m, false);
  if (!hasLinear(m)) {
    require(boot, "no linear or bootstrap candidate sites");
    return;
  }
  Catalog cat(m);
  Builder b(m.getContext());
  auto cuts = cat.layoutCuts;
  SmallVector<Attribute> trace;
  publishRoutes(cat, b);
  for (unsigned round = 1;; ++round) {
    auto regions = cat.regions;
    std::vector<size_t> ids;
    if (independent) {
      for (auto [i,r] : llvm::enumerate(regions)) {
        std::vector<RouteDepthCut> local;
        for (auto& v:cat.layoutCuts) if(v.sites.front()==i) local.push_back({{0},v.limit});
        auto p = selectSubgroupRoutes({r}, cat.sizes, cat.eta, local);
        ids.push_back(p.selected[0]);
        r.candidates = {r.candidates[ids.back()]};
      }
    }
    auto p = selectSubgroupRoutes(regions, cat.sizes, cat.eta, cuts);
    if (!independent) ids = p.selected;
    for (auto [i, op] : llvm::enumerate(cat.ops)) {
      op->setAttr("selected", routeAttr(b, cat.regions[i].candidates[ids[i]]));
      op->setAttr("route_charge",chargeRecord(b,p,i,cat.eta));
    }
    auto violations = analyze(m, false);
    trace.push_back(roundRecord(b,ids,violations));
    if (violations.empty()) {
      m->setAttr("snabf.route_rounds",b.getArrayAttr(trace));
      m->setAttr("snabf.inventory", inventory(b, p));
      m->setAttr("snabf.solve_rounds", b.getI64IntegerAttr(round));
      m->setAttr("snabf.arm",
                 b.getStringAttr(independent ? "independent" : "joint"));
      return;
    }
    for (auto &v : violations) {
      require(!independent,
              "independent selection exceeds the sink rotation budget");
      require(!llvm::any_of(cuts,
                            [&](auto &c) {
                              return c.sites == v.sites && c.limit == v.limit;
                            }),
              "inconsistent repeated path witness");
      cuts.push_back(std::move(v));
    }
  }
}
namespace {
// Level changes are materialized on the structured SSA before Pass 1 pricing.
// Native HEIR type attributes here are import provenance, not the schedule.
bool schedule(ModuleOp m, bool check) {
  bool ok = true;
  m.walk([&](Operation *o){
    if (!ok || o->getNumResults()!=1 || !isa<CtType>(o->getResult(0).getType())) return;
    auto v=o->getResult(0); auto t=cast<CtType>(v.getType()); int64_t l;
    if(isa<SourceOp>(o)) {
      auto profile=SymbolTable::lookupNearestSymbolFrom<ProfileOp>(o,t.getProfile());
      l=int64_t(profile.getQ().size())-1;
    }
    else {
      l=cast<CtType>(o->getOperand(0).getType()).getLevel();
      if (auto c = dyn_cast<ConvertOp>(o)) {
        auto p = SymbolTable::lookupNearestSymbolFrom<ProfileOp>(o, t.getProfile());
        if (!p) { ok = false; return; }
        int64_t cut = int64_t(p.getQ().size()) - 1;
        if (l < cut || (l != cut && !c->getAttrOfType<UnitAttr>("snabf.cut_align"))) { ok = false; return; }
        l = cut;
      }
      else if(isa<LintransOp>(o)) --l;
      else if(isa<PolyEvalOp>(o)) l-=field<IntegerAttr>(field<DictionaryAttr>(o,"poly_selected"),"delta_level").getInt();
      else if(auto s=dyn_cast<StepOp>(o);s && s.getKind()=="ckks.add") l=std::min(l,cast<CtType>(o->getOperand(1).getType()).getLevel());
      else throw ContractError("unscheduled operation in Pass 2 region");
    }
    if(l<0){ok=false;return;}
    if(check) require(l==t.getLevel(),"stale materialized Pass 2 RNS level");
    else v.setType(CtType::get(m.getContext(),t.getLogical(),t.getN(),l,t.getProfile()));
  });
  return ok;
}
double dagWork(DictionaryAttr c,int64_t inputLevel,bool weighted) {
  double cost=0;
  for(auto a:field<ArrayAttr>(c,"nodes")) {
    auto n=cast<DictionaryAttr>(a);auto op=field<StringAttr>(n,"op").getValue();
    if(op=="input")continue;
    int64_t arity=field<DenseI64ArrayAttr>(n,"args").size();
    // Relative per-active-limb work: multiply, key-switch, rescale, and
    // weighted plaintext dot products are charged separately from key bytes.
    double w=op=="mul" ? 3.0 : op=="linear" ? (arity+1.0) : 0.125*arity;
    auto before=field<IntegerAttr>(n,"depth").getInt()-(op=="mul"||op=="linear");
    cost+=w*(weighted ? 1+inputLevel-before : 1);
  }
  return cost;
}
bool sinkContracts(ModuleOp m,std::string &why) {
  bool ok=true;
  m.walk([&](SinkOp s){auto d=field<BudgetAttr>(s,"budget").getFields();
    if(auto l=s->getAttrOfType<IntegerAttr>("min_output_level")) {
      require(l.getInt()>=0,"negative sink level reserve");
      if(field<IntegerAttr>(d,"level").getInt()<l.getInt()){why="suffix_level";ok=false;}
    }
    if(auto eps=s->getAttrOfType<FloatAttr>("max_approx_error")) {
      require(std::isfinite(eps.getValueAsDouble()) && eps.getValueAsDouble()>=0,"invalid sink precision budget");
      if(field<FloatAttr>(d,"approx_error").getValueAsDouble()>eps.getValueAsDouble()){why="suffix_approximation";ok=false;}
    }
  }); return ok;
}
struct PolyCut { PolyEvalOp site; int64_t maxDepth=INT64_MAX;
  double maxError=std::numeric_limits<double>::infinity(); };
// Necessary conditions, not a heuristic penalty: level at an add is min,
// approximation components at an add are nonnegative, and linear transfer
// multiplies that component by the operator infinity norm.
SmallVector<PolyCut> traceViolations(ModuleOp m) {
  SmallVector<PolyCut> cuts;
  std::function<void(Value,int64_t,double)> back=[&](Value v,int64_t reserve,double eps){
    Operation *o=v.getDefiningOp(); if(!o || isa<SourceOp>(o))return;
    if(auto p=dyn_cast<PolyEvalOp>(o)){cuts.push_back({p,reserve<0?INT64_MAX:p.getInput().getType().getLevel()-reserve,eps});return;}
    if(auto c=dyn_cast<ConvertOp>(o)){back(c.getInput(),reserve,eps);return;}
    if(auto l=dyn_cast<LintransOp>(o)){
      auto n=norm(preparedWeights(l.getTable(),l.getDiagonals()),true);
      auto childEps=n==0?INFINITY:std::nextafter(eps/n,INFINITY);
      back(l.getInput(),reserve<0?-1:reserve+1,childEps);return;
    }
    if(auto s=dyn_cast<StepOp>(o);s && s.getKind()=="ckks.add")
      for(auto a:s.getInputs())back(a,reserve,eps);
  };
  m.walk([&](SinkOp s){auto d=field<BudgetAttr>(s,"budget").getFields();
    int64_t reserve=-1;double eps=INFINITY;
    if(auto l=s->getAttrOfType<IntegerAttr>("min_output_level");l && field<IntegerAttr>(d,"level").getInt()<l.getInt())reserve=l.getInt();
    if(auto e=s->getAttrOfType<FloatAttr>("max_approx_error");e && field<FloatAttr>(d,"approx_error").getValueAsDouble()>e.getValueAsDouble())eps=e.getValueAsDouble();
    if(reserve>=0 || std::isfinite(eps))back(s.getInput(),reserve,eps);
  });return cuts;
}
}

void planModule(ModuleOp m,bool independent) {
  (void)affineMode(m);  // Diagnose invalid requests even on an early exit.
  if(!m->hasAttr("snabf.activation_sources")){selectRoutes(m,independent);return;}
  refreshPolynomials(m,false);
  Builder b(m.getContext()); SmallVector<PolyEvalOp> polys; SmallVector<LintransOp> lins;
  m.walk([&](PolyEvalOp p){polys.push_back(p);}); m.walk([&](LintransOp l){lins.push_back(l);});
  require(!polys.empty() && !lins.empty(),"empty Pass 2 graph");
  std::vector<SmallVector<DictionaryAttr>> choices;
  for(auto p:polys){auto &cs=choices.emplace_back();for(auto a:field<ArrayAttr>(p,"poly_frontier"))cs.push_back(cast<DictionaryAttr>(a));
    llvm::stable_sort(cs,[](auto a,auto b){return dagWork(a,0,false)<dagWork(b,0,false);});}
  struct Memo {SmallVector<DictionaryAttr> routes, contexts, charges; SmallVector<ArrayAttr> catalogs;
               DictionaryAttr inventory; Attribute rounds, arm, trace;};
  std::map<std::vector<int64_t>,Memo> cache;
  std::set<std::vector<int64_t>> failedRoutes;
  std::map<std::vector<std::string>,PolyCut> feedback;
  double best=std::numeric_limits<double>::infinity(); SmallVector<DictionaryAttr> selected,winner; SmallVector<Attribute> events;
  auto replay=[&](const Memo &memo) {for(auto [idx,l]:llvm::enumerate(lins)) {
      l->setAttr("selected",memo.routes[idx]); l->setAttr("candidates",memo.catalogs[idx]);
      l->setAttr("route_context",memo.contexts[idx]); l->setAttr("route_charge",memo.charges[idx]);}
    m->setAttr("snabf.route_rounds",memo.trace);
    m->setAttr("snabf.inventory",memo.inventory);m->setAttr("snabf.solve_rounds",memo.rounds);m->setAttr("snabf.arm",memo.arm);
    require(analyze(m,false).empty(),"cached routes violate SSA path");};
  auto event=[&](StringRef why,ArrayRef<int64_t> levels,std::optional<double> score) {
    NamedAttrList e;SmallVector<Attribute> ids;for(auto c:selected)ids.push_back(field<StringAttr>(c,"candidate_id"));
    e.set("candidates",b.getArrayAttr(ids));e.set("reason",b.getStringAttr(why)); e.set("linear_input_levels",b.getDenseI64ArrayAttr(levels));
    if(score) e.set("objective",b.getF64FloatAttr(*score)); events.push_back(e.getDictionary(m.getContext()));};
  auto context=[&](size_t index){std::vector<std::string> key{std::to_string(index)};
    // Ancestor choices fix this site's input level. Later choices need not
    // match: the suffix traced here contains only straight linear/add edges.
    for(size_t j=0;j<index;++j)key.push_back(field<StringAttr>(selected[j],"candidate_id").str());return key;};
  std::function<void(size_t,double)> visit=[&](size_t pos,double lower){
    if(lower>best) return;  // All remaining primitive and route costs are nonnegative.
    if(pos!=polys.size()) {
      for(auto c:choices[pos]){polys[pos]->setAttr("poly_selected",c);selected.push_back(c);
        auto it=feedback.find(context(pos));
        if(it!=feedback.end() && (field<IntegerAttr>(c,"delta_level").getInt()>it->second.maxDepth || field<FloatAttr>(c,"activation_error").getValueAsDouble()>it->second.maxError))
          event("upstream_cut",{},{});
        else visit(pos+1,lower+dagWork(c,0,false));selected.pop_back();}return;
    }
    if(!schedule(m,false)){event("suffix_level",{},{});return;}
    std::vector<int64_t> levels;for(auto l:lins)levels.push_back(l.getInput().getType().getLevel());
    if(failedRoutes.contains(levels)){event("cached_rotation_conflict",levels,{});return;}
    bool fresh=!cache.contains(levels);
    if(fresh){
      try {selectRoutes(m,independent);} catch(const std::runtime_error &ex) {
        if(StringRef(ex.what())!="no optimal subgroup routing solution" && StringRef(ex.what())!="independent selection exceeds the sink rotation budget") throw;
        failedRoutes.insert(levels);event("rotation_path_infeasible",levels,{});return;
      }
      Memo mem;for(auto l:lins){mem.routes.push_back(field<DictionaryAttr>(l,"selected"));
        mem.contexts.push_back(field<DictionaryAttr>(l,"route_context"));mem.charges.push_back(field<DictionaryAttr>(l,"route_charge"));
        mem.catalogs.push_back(field<ArrayAttr>(l,"candidates"));}mem.trace=m->getAttr("snabf.route_rounds");
      mem.inventory=field<DictionaryAttr>(m,"snabf.inventory");mem.rounds=m->getAttr("snabf.solve_rounds");mem.arm=m->getAttr("snabf.arm");cache.emplace(levels,mem);
    } else replay(cache.at(levels));
    std::string why;
    if(!sinkContracts(m,why)){event(why,levels,{});
      for(auto cut:traceViolations(m)){auto index=llvm::find(polys,cut.site)-polys.begin();auto &to=feedback[context(index)];
        to.site=cut.site;to.maxDepth=std::min(to.maxDepth,cut.maxDepth);to.maxError=std::min(to.maxError,cut.maxError);
        NamedAttrList rec;rec.set("reason",b.getStringAttr("backward_ssa"));rec.set("site",b.getI64IntegerAttr(index));
        if(to.maxDepth!=INT64_MAX)rec.set("max_depth",b.getI64IntegerAttr(to.maxDepth));
        if(std::isfinite(to.maxError))rec.set("max_activation_error",b.getF64FloatAttr(to.maxError));events.push_back(rec.getDictionary(m.getContext()));}
      return;}
    double score=field<FloatAttr>(cache.at(levels).inventory,"objective").getValueAsDouble();
    for(auto [idx,p]:llvm::enumerate(polys))score+=dagWork(selected[idx],p.getInput().getType().getLevel(),true);
    if(score<best){best=score;winner=selected;event("best_feasible",levels,score);}else event(fresh?"route_repriced":"level_cache_hit",levels,score);
  };
  visit(0,0); require(!winner.empty(),"no joint SiLU/route plan satisfies downstream level, approximation and rotation budgets");
  for(auto [i,p]:llvm::enumerate(polys))p->setAttr("poly_selected",winner[i]);
  require(schedule(m,false),"invalid winning schedule"); std::vector<int64_t> levels;for(auto l:lins)levels.push_back(l.getInput().getType().getLevel());
  replay(cache.at(levels));m->setAttr("snabf.poly_decisions",b.getArrayAttr(events));m->setAttr("snabf.joint_objective",b.getF64FloatAttr(best));
}
void verifyPlan(ModuleOp m) {
  (void)affineMode(m);
  auto boot = evaluateBootstraps(m, true);
  if (!hasLinear(m)) {
    require(boot, "no selected implementation");
    return;
  }
  bool logical=m->hasAttr("snabf.activation_sources");
  if(logical){refreshPolynomials(m,true);require(schedule(m,true),"infeasible Pass 2 schedule");}
  Catalog c(m);
  Builder b(m.getContext());
  std::vector<size_t> ids;
  for (auto [i, op] : llvm::enumerate(c.ops)) {
    auto &r = c.regions[i];
    require(field<IntegerAttr>(op, "site").getInt() == int64_t(i) &&
                alternatives(b, r) == op->getAttr("candidates"),
            "stale candidate catalog");
    auto it = llvm::find_if(r.candidates, [&](auto &x) {
      return routeAttr(b, x) == op->getAttr("selected");
    });
    require(it != r.candidates.end(), "invalid selected tower");
    require(c.contexts[i]==op->getAttr("route_context"),"stale route pricing, key family or layout feedback");
    ids.push_back(it-r.candidates.begin());
  }
  auto p=scoreSelection(c.regions,c.sizes,c.eta,ids);
  for (auto [i,op]:llvm::enumerate(c.ops)) require(op->getAttr("route_charge")==chargeRecord(b,p,i,c.eta),"stale site key/work charge");
  require(inventory(b, p) == m->getAttr("snabf.inventory"),
          "stale physical key inventory or cost");
  require(analyze(m, true).empty(),
          "selected plan exceeds sink rotation budget");
  auto rounds=field<ArrayAttr>(m,"snabf.route_rounds");
  require(rounds.size()>0 && int64_t(rounds.size())==field<IntegerAttr>(m,"snabf.solve_rounds").getInt(),"stale route round count");
  // Only the scratch module receives the intermediate route assignments.
  OwningOpRef<ModuleOp> scratch(cast<ModuleOp>(m->clone()));
  SmallVector<LintransOp> sites; scratch->walk([&](LintransOp op){sites.push_back(op);});
  auto active=c.layoutCuts;
  for(auto [j,a]:llvm::enumerate(rounds)) {
    auto rec=cast<DictionaryAttr>(a); auto picks=field<DenseI64ArrayAttr>(rec,"choices").asArrayRef();
    require(picks.size()==sites.size(),"invalid route round width"); std::vector<size_t> vi;
    for(auto [i,id]:llvm::enumerate(picks)) {
      require(id>=0 && size_t(id)<c.regions[i].candidates.size(),"invalid route round index");
      vi.push_back(id);sites[i]->setAttr("selected",routeAttr(b,c.regions[i].candidates[id]));
    }
    for(const auto& cut:active) {
      size_t depth=0;for(auto k:cut.sites)depth+=rotationDepth(c.regions[k].candidates[vi[k]]);
      require(depth<=cut.limit,"route round violates an active path or layout constraint");
    }
    auto violated=analyze(*scratch,false); bool last=j+1==rounds.size();
    require(rec==roundRecord(b,vi,violated) && last==violated.empty() && (!last || vi==ids),"stale route feedback trace");
    active.insert(active.end(),violated.begin(),violated.end());
  }
  if(logical){std::string why;require(sinkContracts(m,why),"selected Pass 2 plan violates sink contract");
    auto joint=p.objective;m.walk([&](PolyEvalOp op){joint+=dagWork(field<DictionaryAttr>(op,"poly_selected"),op.getInput().getType().getLevel(),true);});
    require(field<FloatAttr>(m,"snabf.joint_objective").getValueAsDouble()==joint,"stale joint cost");}
}
}  // namespace snabf::ir
