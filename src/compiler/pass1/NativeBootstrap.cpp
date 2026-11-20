#include "NativeRouting.h"
#include "RingSwitch.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cmath>
#include <set>

namespace snabf::ir {
namespace {
// Bind the circuit with the residual prime widths it was tested with, not
// just a count of Q/P limbs. Lattigo constructs and checks the actual rings.
bool matchesPeriodicProfile(mlir::heir::lattigo::CKKSParametersLiteralAttr p) {
  if (!p.getQ() || !p.getP() || p.getLogN() != 16 ||
      p.getLogDefaultScale() != 31 || p.getQ().size() != 5 ||
      p.getP().size() != 2)
    return false;
  std::set<int64_t> primes;
  auto near = [&](int64_t q, int bits) {
    return q > 0 && q % (1 << 17) == 1 && primes.insert(q).second &&
           std::abs(std::log2(double(q)) - bits) < 0.01;
  };
  for (auto [i, q] : llvm::enumerate(p.getQ().asArrayRef()))
    if (!near(q, i ? 31 : 40))
      return false;
  for (auto q : p.getP().asArrayRef())
    if (!near(q, 56))
      return false;
  return true;
}
} // namespace
unsigned evaluateBootstraps(ModuleOp m, bool checkOnly) {
  Builder b(m.getContext());
  unsigned count = 0;
  m.walk([&](mlir::heir::lattigo::CKKSBootstrapOp boot) {
    ++count;
    auto cfg = field<DictionaryAttr>(boot, "snabf.boot_costs");
    mlir::heir::lattigo::CKKSParametersLiteralAttr p;
    m.walk([&](mlir::heir::lattigo::CKKSNewParametersFromLiteralOp op) {
      require(!p, "ambiguous bootstrap key family");
      p = op.getParamsLiteral();
    });
    require(p && p.getQ() && p.getLogN() >= 2 && p.getLogN() <= 30,
            "missing configured CKKS literal");
    // The period belongs to the ciphertext producer/argument. Occupancy and
    // a Boolean in a cost table cannot establish equality inside trace orbits.
    Value ct = boot->getOperand(1);
    IntegerAttr period;
    if (auto def = ct.getDefiningOp())
      period = def->getAttrOfType<IntegerAttr>("snabf.slot_period");
    else if (auto a = dyn_cast<BlockArgument>(ct)) {
      auto f = dyn_cast<func::FuncOp>(a.getOwner()->getParentOp());
      if (f && a.getOwner() == &f.getBody().front())
        period = f.getArgAttrOfType<IntegerAttr>(a.getArgNumber(),
                                                 "snabf.slot_period");
    }
    auto integer = [&](StringRef k) {
      return field<IntegerAttr>(cfg, k).getInt();
    };
    auto price = [&](StringRef k) {
      return field<FloatAttr>(cfg, k).getValueAsDouble();
    };
    BootContract c{};
    c.n = 1LL << p.getLogN();
    c.maxLevel = int64_t(p.getQ().size()) - 1;
    c.target = integer("target_n");
    c.beta = integer("multiplicity");
    c.level = field<IntegerAttr>(boot, "snabf_input_level").getInt();
    c.live = field<IntegerAttr>(boot, "snabf_live_slots").getInt();
    c.period = period ? period.getInt() : 0;
    c.layout = field<StringAttr>(cfg, "layout_contract").str();
    c.layoutAdmitted = field<BoolAttr>(cfg, "layout_verified").getValue();
    c.down = price("down_cost");
    c.up = price("up_cost");
    c.prepare = price("layout_cost");
    c.keyVariance = price("key_variance");
    if (auto a = cfg.getAs<IntegerAttr>("min_small_level"))
      c.minSmallLevel = a.getInt();
    if (auto a = cfg.getAs<FloatAttr>("incoming_variance"))
      c.incomingVariance = a.getValueAsDouble();
    if (auto a = cfg.getAs<FloatAttr>("variance_ceiling"))
      c.varianceCeiling = a.getValueAsDouble();
    auto w = field<DenseF64ArrayAttr>(cfg, "gadget_weights").asArrayRef();
    c.weights.assign(w.begin(), w.end());
    auto bins = field<DenseF64ArrayAttr>(cfg, "downstream_bins").asArrayRef();
    require(bins.size() % 2 == 0, "expected variance/cost pairs");
    for (size_t i = 0; i < bins.size(); i += 2)
      c.downstreamBins.emplace_back(bins[i], bins[i + 1]);
    for (auto a : field<ArrayAttr>(cfg, "boot_costs")) {
      auto row = cast<DictionaryAttr>(a);
      require(c.bootCosts
                  .emplace(
                      std::make_pair(field<IntegerAttr>(row, "n").getInt(),
                                     field<IntegerAttr>(row, "level").getInt()),
                      field<FloatAttr>(row, "cost").getValueAsDouble())
                  .second,
              "duplicate bootstrap calibration");
    }
    auto d = evaluateRingSwitch(c);
    NamedAttrList out;
    auto i64 = [&](StringRef k, int64_t v) {
      out.set(k, b.getI64IntegerAttr(v));
    };
    auto real = [&](StringRef k, std::optional<double> v) {
      if (v)
        out.set(k, b.getF64FloatAttr(*v));
    };
    out.set("enabled", b.getBoolAttr(d.enabled));
    out.set("reason", b.getStringAttr(d.reason));
    auto kernel = boot->getAttrOfType<StringAttr>("snabf.bootstrap_kernel");
    bool supported = kernel && kernel == "periodic-n15-v1" && d.enabled &&
                     c.layout == "trace-periodic" && c.beta == 1 &&
                     c.n == 65536 && c.target == 32768 && c.period >= 8 &&
                     c.level >= 1 && matchesPeriodicProfile(p);
    out.set("execution",
            b.getStringAttr(supported ? "periodic-n15-v1" : "planning_only"));
    out.set("layout_contract", b.getStringAttr(c.layout));
    i64("source_n", c.n);
    i64("target_n", c.target);
    i64("input_level", c.level);
    i64("min_small_level", c.minSmallLevel);
    if (c.level >= 1)
      i64("small_level", c.level - 1);
    i64("live_slots", c.live);
    i64("source_period", c.period);
    i64("beta_pack", c.beta);
    real("rs_variance", d.rsVariance);
    real("input_variance", c.incomingVariance);
    real("output_variance", d.outputVariance);
    real("variance_ceiling", c.varianceCeiling);
    real("down_cost", c.down);
    real("one_up_cost", c.up);
    real("layout_cost", c.prepare);
    real("downstream_penalty", d.downstreamPenalty);
    real("savings", d.savings);
    real("penalty", d.penalty);
    real("margin", d.margin);
    auto rec = out.getDictionary(m.getContext());
    if (checkOnly)
      require(rec == boot->getAttr("snabf.ring_switch"),
              "stale ring-switch decision");
    else
      boot->setAttr("snabf.ring_switch", rec);
  });
  return count;
}

// The outlined HEIR function has one encrypted argument and returns precisely
// its bootstrap result. Mixed graphs still go through the regular CKKS path.
DictionaryAttr executableBootstrap(ModuleOp m) {
  using namespace mlir::heir::lattigo;
  Builder b(m.getContext());
  CKKSBootstrapOp site;
  CKKSParametersLiteralAttr profile;
  unsigned total = 0;
  bool structured = false;
  m.walk([&](Operation *o) {
    if (auto op = dyn_cast<CKKSBootstrapOp>(o)) {
      ++total;
      if (op->hasAttr("snabf.bootstrap_kernel"))
        site = op;
    }
    if (isa<LintransOp, PolyEvalOp, ConvertOp, ProfileOp>(o))
      structured = true;
  });
  if (!site)
    return {};
  require(total == 1 && !structured,
          "unsupported mixed ring-bootstrap lowering");
  evaluateBootstraps(m, true);
  auto rec = field<DictionaryAttr>(site, "snabf.ring_switch");
  require(field<StringAttr>(rec, "execution") == "periodic-n15-v1",
          "selected ring-switched bootstrap has no executable lowering");
  auto func = site->getParentOfType<func::FuncOp>();
  require(func && func.getBody().hasOneBlock() && func.getNumArguments() == 5 &&
              func.getNumResults() == 1 &&
              func.getBody().front().getOperations().size() == 2 &&
              site->getOperand(0) == func.getArgument(0) &&
              site->getOperand(1) == func.getArgument(4) &&
              isa<func::ReturnOp>(func.getBody().front().getTerminator()) &&
              func.getBody().front().getTerminator()->getNumOperands() == 1 &&
              func.getBody().front().getTerminator()->getOperand(0) ==
                  site->getResult(0),
          "unrecognized bootstrap ABI");
  m.walk([&](CKKSNewParametersFromLiteralOp op) {
    auto config = op->getParentOfType<func::FuncOp>();
    require(!profile && config &&
                config.getName() == func.getName().str() + "__configure",
            "unbound bootstrap configuration");
    profile = op.getParamsLiteral();
  });
  require(bool(profile), "missing bootstrap literal");
  NamedAttrList d(rec);
  d.set("function", b.getStringAttr(func.getName()));
  d.set("q", b.getDenseI64ArrayAttr(profile.getQ().asArrayRef()));
  d.set("p", b.getDenseI64ArrayAttr(profile.getP().asArrayRef()));
  d.set("log_scale", b.getI64IntegerAttr(profile.getLogDefaultScale()));
  return d.getDictionary(m.getContext());
}
} // namespace snabf::ir
