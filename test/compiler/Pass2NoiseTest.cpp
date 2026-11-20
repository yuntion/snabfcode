#include "pass2/NoiseTransfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace snabf::pass2;
using Kind = NoisePrimitive;
using Maybe = std::optional<PrimitiveResidual>;

void check(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}
void tight(double actual, double expected, const char* what) {
  check(actual >= expected && actual <= expected + std::abs(expected) * 1e-12, what);
}
template<class Exception = std::invalid_argument, class F> void rejects(F f) {
  try { f(); } catch (const Exception&) { return; }
  throw std::runtime_error("expected rejection");
}

// Analytic fixtures only. Explicitly exact operations are part of these toy
// models, not claims about any HEIR profile, measurements, or actual CKKS.
constexpr char event[] = "fixture:joint-G";
PrimitiveResidual residual(double b, double s, double a, double units = 1) {
  return {event, {b, s, a}, units};
}
ValidatedNoiseProfile fixture() {
  return {"analytic-fixture", event, PlanRange{-2, 2}, 0.03125, 1,
          [](const NoiseSite&) -> Maybe { return residual(0, 0, 0); }};
}
void symbolic(const NoiseTransferResult& r) {
  check(!r.canonical && !r.reason.empty(), "published a numeric certificate or lost diagnostic");
  for (const auto& n : r.nodes) check(!n.decoded, "leaked a partial numeric result");
}

void reusedInput() {
  NumericDAG d{{{"input", {}, {}, 0, 0}, {"add", {0, 0}, {}, 0, 0}}, 1};
  auto p = fixture();
  unsigned sources = 0, adds = 0;
  p.outputScale = 8;
  p.residual = [&](const NoiseSite& s) -> Maybe {
    if (s.primitive == Kind::Source) {
      ++sources;
      return PrimitiveResidual{event, NoiseBounds::fromVarianceProxy(0, 4, 2), 8};
    }
    check(s.primitive == Kind::Add && s.operand == 1, "unexpected alias site");
    ++adds;
    return residual(0, 0.125, 0.125);
  };
  // One Rademacher Z, reused on BOTH input edges and by the add residual:
  // decoded error is (1/4 + 1/4 + 1/8) Z. Its canonical variance is 25.
  auto r = evaluateNoiseTransfer(d, {-1, 1}, 0, 0, &p);
  check(r.canonical && r.reason.empty() && sources == 1 && adds == 1, "lost shared source/site");
  tight(r.nodes[1].decoded.value().centeredSigma, 0.625, "combined independent proxies at fanout");
  tight(r.nodes[0].symbolic.outputSensitivity, 2, "missed an alias in reverse gain");
  tight(r.canonical->varianceProxy, 25, "wrong canonical variance for shared signs");
  tight(r.canonical->absolute, 5, "lost a shared absolute contribution");
  tight(r.canonical->meanBias, 0, "invented mean bias");
  check(r.canonical->validityFailure == *p.validityFailure && r.canonical->commonEvent == event &&
            r.canonical->contractId == p.contractId, "lost or double-counted common event");
}

void complexCrossTerm() {
  NumericDAG d{{{"input", {}, {}, 0, 0}, {"mul", {0, 0}, {}, 0, 1}}, 1};
  auto p = fixture();
  p.residual = [](const NoiseSite& s) -> Maybe {
    if (s.primitive == Kind::Source)
      return residual(0, 1, NoiseBounds::modulusFromComponentBound(1));
    return residual(0, 0, 0);
  };
  auto r = evaluateNoiseTransfer(d, {0, 0}, 1, 0, &p);
  auto out = r.canonical.value();
  check(r.nodes[1].symbolic.productRemainder, "symbolic square missing its cross term");
  // The same source contract admits both laws below. For e=Z(1+i), e^2
  // has imaginary mean 2. For e=Z+iW, independent signs Z,W, Im(e^2)=2ZW
  // has variance 4. Neither result follows from treating e*e as independent.
  std::complex<double> mean{};
  double secondMoment = 0;
  for (double z : {-1., 1.}) {
    auto e = std::complex<double>(z, z);
    mean += e * e * 0.5;
    for (double w : {-1., 1.}) {
      e = {z, w};
      secondMoment += std::pow((e * e).imag(), 2) * 0.25;
    }
  }
  tight(out.meanBias, mean.imag(), "complex product bias lost imaginary component");
  tight(out.varianceProxy, secondMoment, "bounded centered cross term proxy too small");
  tight(out.absolute, 2, "component envelope used as a modulus");
}

PolyCandidate reconstruction() {
  PolyCandidate c;
  // P = x/2 + x * core, core = x^2/4 - 3*x + 4*x + 1/8.
  c.dag = {{{"input", {}, {}, 0, 0}, {"mul", {0, 0}, {}, 0, 1},
            {"weighted_sum", {1, 0, 0}, {0.25, -3, 4}, 0.125, 2},
            {"linear", {0}, {0.5}, 0, 1}, {"mul", {0, 2}, {}, 0, 3},
            {"add", {3, 4}, {}, 0, 3}}, 5};
  c.inputLevel = 4; c.outputLevel = 1; c.idealInput = {2, 2};
  c.approximationError = 0.125;  // Reference for this fixture is P - 1/8.
  // These are deliberately NOT inputs to the stochastic calculation.
  c.cost.ctMults = c.nominalPSMults = 999; c.outputError = 99; c.outputGain = 42;
  c.noise.canonicalVarianceProxy = 123;
  return c;
}

void fullDAG() {
  auto c = reconstruction();
  auto p = fixture(); p.outputScale = 16;
  p.residual = [](const NoiseSite& s) -> Maybe {
    // Three deterministic errors on alignment COPIES; source itself is exact.
    if (s.primitive == Kind::Drop) {
      if (s.node == 2 && s.operand == 1) return residual(1, 0, 1);  // actual -1 before weight -3
      if (s.node == 4 && s.operand == 0) return residual(0.5, 0, 0.5);
      if (s.node == 5 && s.operand == 0) return residual(0.25, 0, 0.25);
    }
    return residual(0, 0, 0);
  };
  auto r = evaluateNoiseTransfer(c, &p);
  const auto& k = r.cost;
  check(k.depth == 3 && k.ctMults == 2 && k.relins == 2 && k.ptMults == 4 && k.rescales == 4 &&
            k.additions == 4 && k.alignmentDrops == 6, "wrong full materialized DAG counts");
  const auto& by = k.byDepth;
  check(by.size() == 4 && by[0].ctSquares == 1 && by[0].rescales == 2 && by[2].ctProducts == 1 &&
            by[1].ptMults == 3 && by[1].ctAdds == 2 && by[2].ptAdds == 1 && by[3].ctAdds == 1 &&
            by[0].levelReductions == std::vector<unsigned>{0, 2, 1} &&
            by[1].levelReductions == std::vector<unsigned>{0, 0, 1} && by[2].alignmentDrops == 1,
        "wrong depth, kind or span for materialized work");
  const auto& sites = r.nodes[2].sites;
  const std::array kinds{Kind::Drop, Kind::Drop, Kind::PtMul, Kind::PtMul, Kind::Add, Kind::PtMul,
                         Kind::Add, Kind::Rescale, Kind::AddPlain};
  check(sites.size() == kinds.size(), "weighted sum omitted or invented a primitive");
  for (std::size_t i = 0; i < sites.size(); ++i)
    check(sites[i].primitive == kinds[i], "wrong materialized primitive order");
  check(sites[0].operand == 1 && sites[1].operand == 2 && sites[0].inputLevel == 4 &&
            sites[0].outputLevel == 3 && sites[7].inputLevel == 3 && sites[7].outputLevel == 2 &&
            sites[8].inputLevel == 2 && sites[8].outputLevel == 2, "wrong site key/scale boundary");
  check(r.nodes[4].sites[0].inputLevel - r.nodes[4].sites[0].outputLevel == 2 &&
            r.nodes[4].sites.size() == 4, "split a multi-prime drop into multiple residual calls");
  tight(r.nodes[2].decoded.value().meanBias, 3, "drop was not weighted before accumulation");
  tight(r.nodes[3].decoded.value().absolute, 0, "drop corrupted bypass source");
  tight(r.nodes[4].decoded.value().meanBias, 9.0625, "omitted the cross term of a dropped operand");
  tight(r.nodes[5].decoded.value().meanBias, 9.3125, "missed terminal bypass alignment");
  tight(r.canonical.value().meanBias, 151, "activation approximation not added exactly once");
  tight(r.canonical->absolute, 151, "wrong canonical envelope");
  tight(r.canonical->varianceProxy, 0, "deterministic approximation inflated centered proxy");
  check(r.fullApproximationError == c.approximationError, "lost decoded approximation record");
}

void primitiveUnits() {
  NumericDAG d{{{"input", {}, {}, 0, 0}, {"mul", {0, 0}, {}, 0, 1},
                {"weighted_sum", {1, 1}, {2, 1}, 0.125, 2}}, 2};
  auto p = fixture(); p.outputScale = 16;
  p.residual = [](const NoiseSite& s) -> Maybe {
    // eta = a*(1+Z) for the same sign Z at every site. The local contracts
    // below are in DIFFERENT canonical scales, including the pre-rescale ones.
    auto inUnits = [](double a, double delta) { return residual(a, a, 2*a, delta); };
    switch (s.primitive) {
      case Kind::Source: return residual(0, 0, 0, 8);
      case Kind::Mul: return inUnits(16, 64);         // 1/4 decoded
      case Kind::Relin: return inUnits(32, 64);       // 1/2, same product scale
      case Kind::Rescale: return s.node == 1 ? inUnits(1, 8) : inUnits(2, 16);
      case Kind::PtMul: return inUnits(s.operand == 0 ? 16 : 32, 64);
      case Kind::Add: return inUnits(16, 64);         // 1/4, before rescale
      case Kind::AddPlain: return inUnits(4, 16);     // 1/4, after rescale
      case Kind::Drop: throw std::runtime_error("invented drop");
    }
    throw std::runtime_error("unexpected site");
  };
  auto r = evaluateNoiseTransfer(d, {1, 1}, 2, 0.125, &p);
  tight(r.nodes[1].decoded.value().meanBias, 0.875, "mul/relin/rescale residual not normalized");
  tight(r.nodes[2].decoded.value().centeredSigma, 4, "lost dependent primitive residuals");
  auto out = r.canonical.value();
  tight(out.meanBias, 66, "incorrect decoded-to-canonical mean/approximation");
  tight(out.varianceProxy, 4096, "canonical variance must use squared physical scale");
  tight(out.absolute, 130, "wrong a.s. canonical complex envelope");
}

void missingContracts() {
  auto c = reconstruction();
  auto noCrypto = evaluateNoiseTransfer(c);
  symbolic(noCrypto);
  check(noCrypto.nodes.size() == c.dag.nodes.size() && noCrypto.cost.ptMults == 4 &&
            noCrypto.fullApproximationError == c.approximationError, "lost symbolic audit");
  for (const auto& node : noCrypto.nodes) for (const auto& missing : node.sites) {
    auto p = fixture();
    p.residual = [&](const NoiseSite& s) -> Maybe {
      if (s.node == missing.node && s.primitive == missing.primitive && s.operand == missing.operand)
        return std::nullopt;
      return residual(0, 0, 0);
    };
    auto r = evaluateNoiseTransfer(c, &p);
    symbolic(r);
    check(r.reason.find("node " + std::to_string(missing.node)) != std::string::npos,
          "missing site not diagnosed");
  }
  auto incomplete = [&](auto edit) {
    auto p = fixture(); edit(p); symbolic(evaluateNoiseTransfer(c, &p));
  };
  incomplete([](auto& p) { p.contractId.clear(); });
  incomplete([](auto& p) { p.commonEvent.clear(); });
  incomplete([](auto& p) { p.idealDomain.reset(); });
  incomplete([](auto& p) { p.idealDomain = PlanRange{-1, 1}; });
  incomplete([](auto& p) { p.outputScale.reset(); });
  incomplete([](auto& p) { p.validityFailure.reset(); });
  incomplete([](auto& p) { p.residual = {}; });
  incomplete([](auto& p) { p.residual = [](const NoiseSite& s) -> Maybe {
    auto r = residual(0, 0, 0);
    if (s.primitive == Kind::Relin) r.commonEvent = "local-event-only";
    return r;
  }; });
  // Zero plaintext gain does not license omitting its local encoding/mul error.
  c.dag.nodes[2].weights[1] = 0;
  auto p = fixture();
  p.residual = [](const NoiseSite& s) -> Maybe {
    if (s.node == 2 && s.primitive == Kind::PtMul && s.operand == 1) return std::nullopt;
    return residual(0, 0, 0);
  };
  symbolic(evaluateNoiseTransfer(c, &p));
}

void invalidInputs() {
  auto c = reconstruction();
  auto corrupt = [&](auto edit) {
    auto bad = c; edit(bad); rejects([&] { (void)evaluateNoiseTransfer(bad); });
  };
  corrupt([](auto& a) { a.dag.nodes[2].inputs[0] = 2; });
  corrupt([](auto& a) { a.dag.nodes[2].weights.clear(); });
  corrupt([](auto& a) { a.dag.nodes[1].depth = 0; });
  corrupt([](auto& a) { a.dag.nodes[2].op = "unknown"; });
  corrupt([](auto& a) { a.dag.output = 2; });
  corrupt([](auto& a) { a.inputLevel = 1; });
  corrupt([](auto& a) { a.idealInput = {2, 1}; });
  corrupt([](auto& a) { a.approximationError = -1; });
  rejects([] { (void)NoiseBounds{0, 0, std::numeric_limits<double>::infinity()}; });
  rejects([] { (void)NoiseBounds::fromVarianceProxy(0, -1, 1); });
  auto p = fixture(); p.validityFailure = 1;
  rejects([&] { (void)evaluateNoiseTransfer(c, &p); });
  p = fixture(); p.outputScale = 0;
  rejects([&] { (void)evaluateNoiseTransfer(c, &p); });
  p = fixture(); p.residual = [](const NoiseSite&) -> Maybe { return residual(0, 0, 0, 0); };
  rejects([&] { (void)evaluateNoiseTransfer(c, &p); });
  p.residual = [](const NoiseSite&) -> Maybe {
    auto r = residual(0, 0, 0); r.bounds.absolute = std::numeric_limits<double>::quiet_NaN(); return r;
  };
  rejects([&] { (void)evaluateNoiseTransfer(c, &p); });
  NumericDAG source{{{"input", {}, {}, 0, 0}}, 0};
  p = fixture(); p.residual = [](const NoiseSite&) -> Maybe {
    return residual(0, std::numeric_limits<double>::max(), 1);
  };
  rejects<std::overflow_error>([&] { (void)evaluateNoiseTransfer(source, {0, 0}, 0, 0, &p); });
  p.residual = [](const NoiseSite&) -> Maybe {
    return residual(0, std::numeric_limits<double>::denorm_min(), 1);
  };
  check(evaluateNoiseTransfer(source, {0, 0}, 0, 0, &p).canonical.value().varianceProxy > 0,
        "underflow silently turned positive variance into exact zero");
}

// Optional integration check when linking the native synthesizer/interval
// libraries; the rest of this test links with only NoiseTransfer.cpp.
#ifdef SNABF_NOISE_TEST_WITH_SYNTHESIS
void nativeCandidates() {
  DPOptions o;
  o.inputLevel = o.maxDepth = 10;
  o.degrees = {3}; o.keepAllFeasible = true;
  auto plans = synthesizePolynomials("silu", {-1, 1}, o);
  check(plans.certificationFailures.empty() && !plans.allFeasible.empty(), "missing synthesized DAGs");
  auto p = fixture(); p.outputScale = 16;
  for (const auto& c : plans.allFeasible) {
    auto symbolicPlan = evaluateNoiseTransfer(c);
    symbolic(symbolicPlan);
    auto r = evaluateNoiseTransfer(c, &p);
    const auto k = costDAG(c.dag);
    const auto& a = r.cost;
    check(a.depth == k.depth && a.ctMults == k.ctMults && a.ptMults == k.ptMults && a.relins == k.relins &&
              a.additions == k.additions && a.rescales == k.rescales && a.alignmentDrops == k.alignmentDrops &&
              a.byDepth.size() == k.byDepth.size(), "emission disagrees with native costDAG");
    for (std::size_t i = 0; i < a.byDepth.size(); ++i) {
      const auto &u = a.byDepth[i], &v = k.byDepth[i];
      check(u.ctProducts == v.ctProducts && u.ctSquares == v.ctSquares && u.ptMults == v.ptMults &&
                u.relins == v.relins && u.rescales == v.rescales && u.ctAdds == v.ctAdds &&
                u.ptAdds == v.ptAdds && u.alignmentDrops == v.alignmentDrops &&
                u.levelReductions == v.levelReductions, "emission disagrees with native per-depth work");
    }
    tight(r.canonical.value().meanBias, c.approximationError * *p.outputScale, "lost native approximation");
    tight(r.canonical->varianceProxy, 0, "approximation counted as random error");
  }
}
#endif

}  // namespace

int main() {
  try {
    reusedInput(); complexCrossTerm(); fullDAG(); primitiveUnits(); missingContracts(); invalidInputs();
#ifdef SNABF_NOISE_TEST_WITH_SYNTHESIS
    nativeCandidates();
#endif
    std::cout << "Pass2NoiseTest: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Pass2NoiseTest: " << e.what() << '\n';
    return 1;
  }
}
