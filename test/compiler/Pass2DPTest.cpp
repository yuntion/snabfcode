#include "pass2/CompositePolyDP.h"
#include "pass2/PolyTemplates.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace {
using namespace snabf::pass2;

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F f) {
  try { f(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("expected invalid_argument");
}
void clean(const DPResult& result) {
  if (!result.certificationFailures.empty()) throw std::runtime_error(result.certificationFailures.front());
}
bool contains(PlanRange range, long double x, long double tol = 0) {
  return range.lo - tol <= x && x <= range.hi + tol;
}
bool same(PlanRange a, PlanRange b) { return a.lo == b.lo && a.hi == b.hi; }
long double horner(const std::vector<double>& coeffs, long double x) {
  long double y = 0;
  for (std::size_t j = coeffs.size(); j-- > 0;) y = y * x + coeffs[j];
  return y;
}
long double derivative(const std::vector<double>& coeffs, long double x) {
  long double y = 0;
  for (std::size_t j = coeffs.size(); --j > 0;) y = y * x + j * coeffs[j];
  return y;
}
long double reference(StageFunction fn, long double x) {
  switch (fn) {
    case StageFunction::SiluDirect: return std::tanh(x / 2) / 2;
    case StageFunction::TanhQuarter: return std::tanh(x / 4);
    case StageFunction::Rational: return x / (1 + x*x);
    case StageFunction::GeluDirect: return std::erf(x / std::sqrt(2.L)) / 2;
  }
  throw std::logic_error("unknown test reference");
}

void testNodeIR() {
  NumericDAG d{{{"input", {}, {}, 0, 0}, {"mul", {0, 0}, {}, 0, 1},
                {"linear", {1, 0}, {-0.75, 0.25}, 1.5, 2},
                {"add", {2, 0}, {}, -0.125, 2}, {"mul", {3, 0}, {}, 0, 3}}, 4};
  auto c = costDAG(d);
  check(evaluateDAG(d, 0.5) == 0.90625, "linear/add constant or operand ordering");
  check(c.depth == 3 && c.ctMults == 2 && c.relins == 2 && c.ptMults == 2 &&
            c.rescales == 3 && c.additions == 4 && c.alignmentDrops == 5,
        "incorrect count of actual work on fan-out DAG");
  const auto& h = c.byDepth;
  check(h.size() == 4 && h[0].ctSquares == 1 && h[0].relins == 1 && h[0].rescales == 1 &&
            h[1].ptMults == 2 && h[1].ctAdds == 1 && h[1].ptAdds == 0 && h[1].rescales == 1 &&
            h[2].ptAdds == 2 && h[2].ctAdds == 1 && h[2].ctProducts == 1 && h[2].rescales == 1 &&
            h[0].alignmentDrops == 3 && h[1].alignmentDrops == 2 &&
            h[0].levelReductions == std::vector<unsigned>({0, 1, 2}),
        "mispriced primitive level, post-rescale add_plain, or batched alignment");
  auto corrupt = [&](auto edit) { auto bad = d; edit(bad); rejects([&] { (void)costDAG(bad); }); };
  corrupt([](auto& a) { a.nodes[2].inputs[0] = 2; });
  corrupt([](auto& a) { a.nodes[2].weights.pop_back(); });
  corrupt([](auto& a) { a.nodes[2].weights[0] = std::numeric_limits<double>::quiet_NaN(); });
  corrupt([](auto& a) { a.nodes[1].depth = 0; });
  corrupt([](auto& a) { a.nodes[4].constant = 0.25; });
  corrupt([](auto& a) { a.nodes[3].op = "reconstruct_by_magic"; });
  corrupt([](auto& a) { a.output = 3; });
}

std::vector<long double> interpret(const NumericDAG& d, long double x,
                                   const std::vector<long double>& localErrors = {}) {
  std::vector<long double> values;
  for (const auto& n : d.nodes) {
    long double v = n.constant;
    if (n.op == "input") v = x;
    else if (n.op == "mul") v = values.at(n.inputs[0]) * values.at(n.inputs[1]);
    else for (std::size_t i = 0; i < n.inputs.size(); ++i)
      v += (n.op == "linear" ? n.weights.at(i) : 1) * values.at(n.inputs[i]);
    if (!localErrors.empty()) v += localErrors.at(values.size());
    values.push_back(v);
  }
  return values;
}

void verifyCandidate(const PolyCandidate& p, unsigned samples) {
  const auto& templates = templatesFor(p.activation);
  const auto spec = std::find_if(templates.begin(), templates.end(),
                                 [&](const auto& t) { return t.id == p.templateName; });
  check(spec != templates.end() && spec->stages.size() == p.stages.size(), "lost template");
  const auto& d = p.dag;
  const auto& out = d.nodes.at(d.output);
  check(out.op == "add" && out.inputs.size() == 2 && out.constant == 0, "missing reconstruction add");
  const auto& scale = d.nodes.at(out.inputs[0]);
  const auto& product = d.nodes.at(out.inputs[1]);
  check(scale.op == "linear" && scale.inputs == std::vector<unsigned>{0} && scale.depth == 1 &&
            scale.weights == std::vector<double>{0.5} && product.op == "mul" &&
            product.inputs == std::vector<unsigned>{0, p.stages.back().outputNode},
        "missing original-x bypass or terminal ciphertext multiplication");
  const auto cost = costDAG(d);
  check(p.cost.depth == cost.depth && p.outputLevel + cost.depth == p.inputLevel &&
            p.cost.ctMults == cost.relins && p.cost.rescales == cost.rescales &&
            p.cost.rescales >= cost.depth && p.cost.ptMults == cost.ptMults,
        "active-level or operation count mismatch");
  check(p.noise.nodes.size() == d.nodes.size() && !p.noise.canonicalBias &&
            !p.noise.canonicalVarianceProxy && !p.noise.validityFailure,
        "incomplete symbolic contract or fabricated CKKS certificate");
  double gain = 1, e = p.inputError;
  unsigned consumed = 0, nominal = 0, nominalMults = 0, node = 0;
  auto I = p.idealInput, A = p.actualInput;
  for (const auto& s : p.stages) {
    check(same(s.idealIn, I) && same(s.actualIn, A) && s.errorIn == e && s.inputNode == node,
          "DP omitted previous polynomial state");
    check(s.hullIn.lo == std::min(I.lo, A.lo) && s.hullIn.hi == std::max(I.hi, A.hi), "bad J hull");
    check(s.errorOut >= static_cast<long double>(s.lipschitz)*e + s.residual &&
              s.cumulativeGain >= static_cast<long double>(s.lipschitz)*gain,
          "optimistic forward recurrence");
    e = s.errorOut; gain = s.cumulativeGain; I = s.idealOut; A = s.actualOut;
    node = s.outputNode; consumed += s.actualDepth; nominal += s.nominalDepth; nominalMults += s.nominalPSMults;
  }
  check(consumed + 1 == p.cost.depth && nominal == p.nominalDepth && nominalMults == p.nominalPSMults,
        "nominal PS estimates mixed into actual depth");
  check(p.coreGain == gain && p.outputGain >= p.bypassGain && p.outputError >= p.approximationError,
        "bad whole-activation interface");

  // Independent monomial/Horner and analytic-function checks. Sampling here
  // tests the implemented IR and its contracts, not the validity of a proof.
  constexpr long double tol = 3e-15L;
  for (unsigned i = 0; i <= samples; ++i) {
    const long double x = p.idealInput.lo + (static_cast<long double>(p.idealInput.hi)-p.idealInput.lo)*i/samples;
    const long double dx = i % 2 ? p.inputError : -p.inputError;
    const long double xp = x + dx;
    const auto values = interpret(d, xp);
    long double ideal = x, actual = xp, exactInputPoly = x, dg = 1;
    for (std::size_t k = 0; k < p.stages.size(); ++k) {
      const auto& s = p.stages[k];
      check(contains(s.hullIn, ideal, tol) && contains(s.hullIn, actual, tol), "hull missed sample");
      dg *= derivative(s.coefficients, actual);
      actual = horner(s.coefficients, actual);
      exactInputPoly = horner(s.coefficients, exactInputPoly);
      const auto target = reference(spec->stages[k], ideal);
      check(std::abs(horner(s.coefficients, ideal) - target) <= s.residual + tol, "local residual missed sample");
      ideal = target;
      check(std::abs(actual - values.at(s.outputNode)) <= tol && contains(s.actualOut, actual, tol) &&
                contains(s.idealOut, ideal, tol) && std::abs(actual-ideal) <= s.errorOut + tol,
            "stage schedule, range, or accumulated error mismatch");
    }
    const long double y = p.activation == "silu" ? x/(1+std::exp(-x)) : x*(1+std::erf(x/std::sqrt(2.L)))/2;
    const long double yp = xp * (0.5L + actual);
    check(std::abs(ideal*x + x/2 - y) <= tol && std::abs(yp - values[d.output]) <= tol &&
              std::abs(evaluateDAG(d, static_cast<double>(xp)) - yp) <= tol,
          "reconstruction computes the wrong activation");
    check(contains(p.idealOutput, y, tol) && contains(p.actualOutput, yp, tol) &&
              std::abs(yp-y) <= p.outputError+tol && std::abs(x*(0.5L+exactInputPoly)-y) <= p.approximationError+tol,
          "full output range or deterministic error excludes a sample");
    check(std::abs(0.5L + actual + xp*dg) <= p.outputGain + tol, "bypass missing from derivative gain");
    for (std::size_t k=0; k<values.size(); ++k)
      check(contains(p.noise.nodes[k].range, values[k], tol), "node range excludes a value");
  }
}

void verifyNoise(const PolyCandidate& p) {
  const auto& d = p.dag;
  std::vector<long double> eta(d.nodes.size()), bound;
  for (std::size_t i = 0; i < eta.size(); ++i) eta[i] = (i%2 ? -1.L : 1.L) * (i+1) * 1e-7L;
  const long double x = (p.idealInput.lo + p.idealInput.hi)/2.L;
  const auto clean = interpret(d, x), perturbed = interpret(d, x, eta);
  for (std::size_t i=0; i<d.nodes.size(); ++i) {
    const auto& n = d.nodes[i]; const auto& t = p.noise.nodes[i];
    check(t.operandGains.size() == n.inputs.size() && t.productRemainder == (n.op == "mul"),
          "wrong symbolic DAG transfer shape");
    long double e = std::abs(eta[i]);
    for (std::size_t k=0; k<n.inputs.size(); ++k) e += t.operandGains[k]*bound.at(n.inputs[k]);
    if (t.productRemainder) e += bound[n.inputs[0]] * bound[n.inputs[1]];
    check(std::abs(clean[i] - perturbed[i]) <= e+2e-18L, "symbolic error transfer omitted a contribution");
    check(t.outputSensitivity >= 0 && std::isfinite(t.outputSensitivity), "missing reverse DAG sensitivity");
    bound.push_back(e);
  }
  check(p.noise.nodes.back().outputSensitivity == 1, "wrong sink sensitivity");
}

using IDs = std::set<std::string>;
IDs ids(const std::vector<PolyCandidate>& ps) {
  IDs r; for (const auto& p : ps) check(r.insert(p.id).second, "duplicate plan ID");
  return r;
}

void sameIDs(const IDs& expected, const IDs& actual, const std::string& what) {
  for (const auto& id : expected)
    if (!actual.contains(id)) throw std::runtime_error(what + ": missing " + id);
  for (const auto& id : actual)
    if (!expected.contains(id)) throw std::runtime_error(what + ": unexpected " + id);
}

void testSearch() {
  DPOptions o; o.inputLevel = o.maxDepth = 20; o.keepAllFeasible = true;
  const auto brute = synthesizeSiLU({-1, 1}, o);
  clean(brute);
  const unsigned choices = 2 * o.degrees.size();
  check(brute.allFeasible.size() == choices + choices*choices && brute.prefixDominated == 0 &&
            brute.feasible == brute.allFeasible.size() && brute.dominated + brute.frontier.size() == brute.feasible,
        "stage DP did not expand the complete template/degree/schedule space");
  IDs expectedDAGs;
  std::vector<std::string> suffixes;
  for (unsigned d : o.degrees) for (const char* schedule : {"balanced", "ps"})
    suffixes.push_back(".d" + std::to_string(d) + "." + schedule);
  for (const auto& s : suffixes) {
    expectedDAGs.insert("silu_direct" + s);
    for (const auto& t : suffixes) expectedDAGs.insert("silu_tanh_rational" + s + t);
  }
  sameIDs(expectedDAGs, ids(brute.allFeasible), "Cartesian degree/schedule oracle");
  IDs oracle;
  for (const auto& p : brute.allFeasible) {
    verifyCandidate(p, 16);
    if (std::none_of(brute.allFeasible.begin(), brute.allFeasible.end(), [&](const auto& q) { return dominates(q,p); }))
      oracle.insert(p.id);
    if (p.stages.size() == 1 && p.stages[0].degree <= 5) {
      check(p.cost.depth == (p.stages[0].degree == 3 ? 4u : 5u), "direct degree-3/5 lost a level");
      check(p.nominalPSMults == (p.stages[0].degree == 3 ? 2u : 3u) && p.nominalDepth + 2 == p.cost.depth,
            "wrong paper count or uncharged scalar/terminal level");
    }
  }
  check(oracle.size() > 1, "expected nondominated tradeoffs");
  sameIDs(oracle, ids(brute.frontier), "audit frontier vs pairwise oracle");
  o.keepAllFeasible = false;
  const auto dp = synthesizeSiLU({-1, 1}, o);
  clean(dp);
  check(dp.allFeasible.empty(), "audit records leaked into normal result");
  check(dp.prefixDominated == 0, "prefix dominance must be disabled");
  check(dp.transitions == 2*choices + choices*choices && dp.feasible == expectedDAGs.size(),
        "normal mode did not expand all degree/schedule prefixes");
  sameIDs(oracle, ids(dp.frontier), "normal frontier vs exhaustive pairwise oracle");
  const auto direct = [&](const std::string& schedule) -> const PolyCandidate& {
    for (const auto& p : brute.allFeasible)
      if (p.stages.size() == 1 && p.stages[0].degree == 13 && p.stages[0].schedule == schedule) return p;
    throw std::runtime_error("missing degree-13 schedule");
  };
  const auto& ps = direct("ps"); const auto& balanced = direct("balanced");
  check(ps.cost.ctMults < balanced.cost.ctMults && balanced.cost.depth < ps.cost.depth && !(ps.dag == balanced.dag),
        "PS / balanced did not expose the depth vs work tradeoff");
  verifyNoise(ps); verifyNoise(balanced);
  std::cout << "SiLU: " << brute.feasible << " full DAGs; " << dp.frontier.size() << " Pareto; "
            << dp.prefixDominated << " dominated prefixes; direct degrees 3/5 consume 4/5 levels\n";
}

void testLevelPrices() {
  DPOptions o; o.degrees = {3}; o.composed = false;
  const auto plans = synthesizeSiLU({-1,1}, o); clean(plans);
  check(plans.frontier.size() == 2, "final pruning lost a level-specific price tradeoff");
  const auto& a = plans.frontier[0]; const auto& b = plans.frontier[1];
  check(a.stages[0].schedule == "balanced" && b.stages[0].schedule == "ps", "unexpected cubic schedules");
  check(a.cost.depth == b.cost.depth && a.cost.ctMults == b.cost.ctMults &&
            a.cost.relins == b.cost.relins && a.cost.ptMults > b.cost.ptMults,
        "expected aggregate cost to favor PS");
  // Two legitimate nonnegative price tables: just price ct.mul at d=1,
  // then just price ct.mul at d=2. PS and balanced switch order.
  check(a.cost.byDepth[1].ctProducts > b.cost.byDepth[1].ctProducts &&
            a.cost.byDepth[2].ctProducts < b.cost.byDepth[2].ctProducts &&
            !dominates(a,b) && !dominates(b,a), "discarded cheaper schedule under a valid level price");
  // Same polynomial and depth; one strictly redundant pt multiply and ct add
  // at exactly the same level. This is dominance even with arbitrary prices.
  auto redundant = a;
  auto& half = redundant.dag.nodes[redundant.dag.nodes[redundant.dag.output].inputs[0]];
  half.inputs = {0,0}; half.weights = {0.25,0.25};
  redundant.cost = costDAG(redundant.dag);
  check(dominates(a,redundant) && !dominates(redundant,a) && !dominates(a,a),
        "strictly dominated extra work at the same level was not recognized");
  o.direct = false; o.composed = true;
  const auto twice = synthesizeSiLU({-1,1},o); clean(twice);
  std::set<std::string> prefixes;
  for (const auto& p : twice.frontier) prefixes.insert(p.stages[0].schedule);
  check(prefixes == std::set<std::string>({"balanced", "ps"}) && twice.prefixDominated == 0 && twice.feasible == 4,
        "prefix pruning lost the same level-specific tradeoff before the second stage");
}

void testReferenceErrorUnits() {
  // Nonzero template error is in core units, not already output units.
  const auto big = coreToActivationError({-4,2}, 0.03125, 0.125);
  const auto small = coreToActivationError({-0.125,0.25}, 0.03125, 0.125);
  check(big >= 0.625 && big < 0.625000000000001 && small >= 0.0390625 && small < 0.039062500000001 &&
            coreToActivationError({0,0}, 0.03125, 0.125) == 0, "template residual escaped |x| conversion");
  rejects([] { coreToActivationError({-1,1}, 0, -1); });
}

void testConstraints() {
  DPOptions o; o.inputLevel = 8; o.maxDepth = 4; o.degrees = {3, 5, 3};
  o.composed = false;
  auto shallow = synthesizeSiLU({-1, 1}, o); clean(shallow);
  check(!shallow.frontier.empty() && shallow.depthRejected > 0, "expected feasible direct cubic only");
  for (const auto& p : shallow.frontier) check(p.cost.depth == 4 && p.outputLevel == 4, "incorrect output level");
  o.maxDepth = 16; o.minOutputLevel = 5;
  check(synthesizeSiLU({-1, 1}, o).frontier.empty(), "spent reserved downstream level");
  o.minOutputLevel = 0;
  o.maxError = std::nextafter(shallow.frontier.front().outputError, 0.);
  auto tight = synthesizeSiLU({-1, 1}, o); clean(tight);
  check(!tight.frontier.empty() && tight.errorRejected > 0, "error budget did not change selection");
  for (const auto& p : tight.frontier) check(p.stages.front().degree == 5 && p.outputLevel == 3, "lost precision/depth tradeoff");
  o.maxError = 0;
  check(synthesizeSiLU({-1, 1}, o).frontier.empty(), "zero uniform error unexpectedly feasible");
  o.maxError = std::numeric_limits<double>::infinity(); o.inputError = 0.1;
  o.direct = false; o.composed = true; o.degrees = {3}; o.inputLevel = 15;
  auto perturbed = synthesizeSiLU({-1.1, 0.7}, o); clean(perturbed);
  check(!perturbed.frontier.empty(), "asymmetric perturbed two-stage search failed");
  for (const auto& p : perturbed.frontier) {
    verifyCandidate(p, 64); verifyNoise(p);
    check(p.stages.size() == 2 && !same(p.stages[1].actualIn, p.stages[1].idealIn) &&
              p.outputError > p.approximationError && p.outputGain > 0.5,
          "actual trajectory or input perturbation collapsed into ideal one");
  }
  o.inputError = 0; o.direct = true; o.composed = false;
  const auto gelu = synthesizePolynomials("gelu", {-1, 0.75}, o); clean(gelu);
  check(!gelu.frontier.empty(), "registered GeLU reference missing");
  verifyCandidate(gelu.frontier.front(), 48);
  auto wrongActivation = gelu.frontier.front(); wrongActivation.activation = "silu";
  check(!dominates(wrongActivation,gelu.frontier.front()), "cross-activation dominance");
  o.minOutputLevel = o.inputLevel + 1;
  check(synthesizeSiLU({-1,1},o).frontier.empty(), "impossible downstream reserve underflowed");
  rejects([&] { synthesizePolynomials("unknown", {-1,1},o); });
  rejects([&] { synthesizeSiLU({1,-1},o); });
  o.degrees = {4}; rejects([&] { synthesizeSiLU({-1,1},o); });
  o.degrees = {3}; o.inputError = -0.5; rejects([&] { synthesizeSiLU({-1,1},o); });
  o.inputError = 0; o.maxError = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { synthesizeSiLU({-1,1},o); });
}
}  // namespace

int main() {
  try { testNodeIR(); testLevelPrices(); testReferenceErrorUnits(); testSearch(); testConstraints(); }
  catch (const std::exception& e) { std::cerr << "Pass2DPTest: " << e.what() << '\n'; return 1; }
}
