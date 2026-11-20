#include "NoiseTransfer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace snabf::pass2 {
namespace {

using Kind = NoisePrimitive;
static_assert(std::numeric_limits<double>::is_iec559);
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "NoiseTransfer requires IEEE floating-point semantics (no fast-math)"
#endif

void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}

void bound(double x) { require(std::isfinite(x) && x >= 0, "invalid noise bound"); }
void scale(double x) { require(std::isfinite(x) && x > 0, "invalid physical scale"); }
void interval(PlanRange r) {
  require(std::isfinite(r.lo) && std::isfinite(r.hi) && r.lo <= r.hi,
          "invalid real input interval");
}

// Round outward even on platforms where long double is no wider than double.
// Exact zeros/units are preserved: e.g. an explicitly exact primitive cannot
// introduce noise just because its bound was converted to another scale.
double outward(double x, bool up = true) {
  const double inf = std::numeric_limits<double>::infinity();
  const double y = std::nextafter(x, up ? inf : -inf);
  if (!std::isfinite(x) || !std::isfinite(y))
    throw std::overflow_error("unrepresentable noise transfer bound");
  return y;
}
double sum(double a, double b, bool up = true) {
  if (a == 0 || b == 0) return a == 0 ? b : a;
  return outward(a + b, up);
}
double product(double a, double b, bool up = true) {
  if (a == 0 || b == 0) return 0;
  if (a == 1 || b == 1) return a == 1 ? b : a;
  if (a == -1 || b == -1) return a == -1 ? -b : -a;
  return outward(a * b, up);
}
double quotient(double a, double b) {
  if (a == 0 || b == 1) return a;
  return outward(a / b);
}
double upperSqrt(double x) { return x == 0 || x == 1 ? x : outward(std::sqrt(x)); }

double magnitude(PlanRange x) { return std::max(std::abs(x.lo), std::abs(x.hi)); }
PlanRange rangeSum(PlanRange a, PlanRange b) {
  return {sum(a.lo, b.lo, false), sum(a.hi, b.hi)};
}
PlanRange rangeProduct(PlanRange a, PlanRange b, bool square = false) {
  if (square) {
    double near = a.lo <= 0 && a.hi >= 0 ? 0 : std::min(std::abs(a.lo), std::abs(a.hi));
    double far = magnitude(a);
    return {std::max(0.0, product(near, near, false)), product(far, far)};
  }
  PlanRange r{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  for (double x : {a.lo, a.hi})
    for (double y : {b.lo, b.hi}) {
      r.lo = std::min(r.lo, product(x, y, false));
      r.hi = std::max(r.hi, product(x, y));
    }
  return r;
}

void count(unsigned& value, unsigned by = 1) {
  if (by > std::numeric_limits<unsigned>::max() - value)
    throw std::overflow_error("unrepresentable DAG operation count");
  value += by;
}

const char* name(Kind k) {
  switch (k) {
    case Kind::Source: return "source";
    case Kind::Drop: return "drop";
    case Kind::Mul: return "mul";
    case Kind::Relin: return "relin";
    case Kind::PtMul: return "ptmul";
    case Kind::Add: return "add";
    case Kind::Rescale: return "rescale";
    case Kind::AddPlain: return "add_plain";
  }
  throw std::logic_error("invalid noise primitive");
}

NoiseTransferResult schedule(const NumericDAG& dag, PlanRange x, unsigned inputLevel) {
  interval(x);
  require(dag.output < dag.nodes.size(), "missing DAG output");
  NoiseTransferResult result;
  auto& cost = result.cost;
  result.nodes.reserve(dag.nodes.size());
  for (std::size_t i = 0; i < dag.nodes.size(); ++i) {
    const auto& n = dag.nodes[i];
    const bool source = n.op == "input", mul = n.op == "mul";
    const bool weighted = n.op == "linear" || n.op == "weighted_sum";
    require(source || mul || weighted || n.op == "add", "unsupported materialized DAG op");
    require(source == (i == 0) && (source == n.inputs.empty()), "invalid DAG source or empty operands");
    require(!mul || n.inputs.size() == 2, "mul requires two operands");
    require(weighted ? n.weights.size() == n.inputs.size() : n.weights.empty(), "invalid DAG weights");
    require(std::isfinite(n.constant) && (!(source || mul) || n.constant == 0), "invalid DAG constant");
    for (double w : n.weights) require(std::isfinite(w), "nonfinite weight");
    unsigned parentDepth = 0;
    for (unsigned j : n.inputs) {
      require(j < i, "non-topological DAG");
      parentDepth = std::max(parentDepth, dag.nodes[j].depth);
    }
    const unsigned levels = mul || weighted ? 1 : 0;
    require(parentDepth <= inputLevel && levels <= inputLevel - parentDepth &&
                n.depth == parentDepth + levels, "invalid DAG depth or insufficient levels");
    const unsigned in = inputLevel - parentDepth, out = in - levels;
    if (cost.byDepth.size() <= n.depth) cost.byDepth.resize(std::size_t(n.depth) + 1);
    auto& rec = result.nodes.emplace_back();
    auto emit = [&](Kind k, std::size_t operand, unsigned from, unsigned to) {
      rec.sites.push_back({i, k, operand, from, to});
      auto& work = cost.byDepth[inputLevel - from];
      switch (k) {
        case Kind::Source: break;
        case Kind::Drop: {
          const unsigned span = from - to;
          count(cost.alignmentDrops, span);
          auto& reductions = work.levelReductions;
          if (reductions.size() <= span) reductions.resize(std::size_t(span) + 1);
          count(reductions[span]);
          for (unsigned level = from; level > to; --level)
            count(cost.byDepth[inputLevel - level].alignmentDrops);
          break;
        }
        case Kind::Mul:
          count(cost.ctMults);
          count(n.inputs[0] == n.inputs[1] ? work.ctSquares : work.ctProducts);
          break;
        case Kind::Relin: count(cost.relins); count(work.relins); break;
        case Kind::PtMul: count(cost.ptMults); count(work.ptMults); break;
        case Kind::Add: count(cost.additions); count(work.ctAdds); break;
        case Kind::AddPlain: count(cost.additions); count(work.ptAdds); break;
        case Kind::Rescale: count(cost.rescales); count(work.rescales); break;
      }
    };
    auto& t = rec.symbolic;
    t.productRemainder = mul;
    auto rangeOf = [&](std::size_t k) { return result.nodes[n.inputs[k]].symbolic.range; };
    for (std::size_t k = 0; k < n.inputs.size(); ++k) {
      const unsigned from = inputLevel - dag.nodes[n.inputs[k]].depth;
      if (from != in) emit(Kind::Drop, k, from, in);
    }
    if (source) {
      t.range = x;
      emit(Kind::Source, 0, in, out);
    } else if (mul) {
      auto a = rangeOf(0), b = rangeOf(1);
      t.range = rangeProduct(a, b, n.inputs[0] == n.inputs[1]);
      t.operandGains = {magnitude(b), magnitude(a)};
      emit(Kind::Mul, 0, in, in);
      emit(Kind::Relin, 0, in, in);
    } else {
      t.range = {0, 0};
      for (std::size_t k = 0; k < n.inputs.size(); ++k) {
        const double w = weighted ? n.weights[k] : 1;
        t.operandGains.push_back(std::abs(w));
        t.range = rangeSum(t.range, rangeProduct({w, w}, rangeOf(k)));
        if (weighted) emit(Kind::PtMul, k, in, in);
        if (k != 0) emit(Kind::Add, k, in, in);
      }
    }
    if (levels) emit(Kind::Rescale, 0, in, out);
    if (n.constant != 0) {
      t.range = rangeSum(t.range, {n.constant, n.constant});
      emit(Kind::AddPlain, 0, out, out);
    }
  }
  std::vector<bool> live(dag.nodes.size());
  live[dag.output] = true;
  result.nodes[dag.output].symbolic.outputSensitivity = 1;
  for (std::size_t i = dag.nodes.size(); i-- > 0;) {
    require(live[i], "dead node in materialized DAG");
    const auto& t = result.nodes[i].symbolic;
    const auto& parents = dag.nodes[i].inputs;
    for (std::size_t k = 0; k < parents.size(); ++k) {
      live[parents[k]] = true;
      auto& g = result.nodes[parents[k]].symbolic.outputSensitivity;
      g = sum(g, product(t.outputSensitivity, t.operandGains[k]));
    }
  }
  cost.depth = dag.nodes[dag.output].depth;
  return result;
}

// Holder on the conditional law: SIGMAS add for arbitrary dependence,
// including aliases, shared randomness and residuals correlated with inputs.
NoiseBounds add(NoiseBounds a, NoiseBounds b) {
  return {sum(a.meanBias, b.meanBias), sum(a.centeredSigma, b.centeredSigma),
          sum(a.absolute, b.absolute)};
}
NoiseBounds amplify(NoiseBounds a, double gain) {
  return {product(a.meanBias, gain), product(a.centeredSigma, gain), product(a.absolute, gain)};
}
NoiseBounds multiply(NoiseBounds a, NoiseBounds b, const NodeTransfer& t) {
  // Real ideal a0,b0 are constants under the fixed-input conditional law.
  // Error = b0*ea + a0*eb + ea*eb. For |ea*eb| <= R on G, BOTH real
  // components lie in [-R,R]. Their means have magnitude <= R; Hoeffding's
  // lemma after centering gives sigma R (range width 2R), even for a square.
  const double r = product(a.absolute, b.absolute);
  const double crossSigma = a.centeredSigma == 0 && b.centeredSigma == 0 ? 0 : r;
  return add(add(amplify(a, t.operandGains[0]), amplify(b, t.operandGains[1])),
             {r, crossSigma, r});
}

std::string unavailable(const ValidatedNoiseProfile* p, PlanRange x) {
  if (!p) return "no validated external primitive contract";
  if (p->contractId.empty() || p->commonEvent.empty()) return "missing contract or common validity event ID";
  if (!p->idealDomain) return "missing validated ideal input domain";
  interval(*p->idealDomain);
  if (p->idealDomain->lo > x.lo || p->idealDomain->hi < x.hi) return "input outside validated ideal domain";
  if (!p->validityFailure) return "missing common-event failure bound";
  require(std::isfinite(*p->validityFailure) && *p->validityFailure >= 0 && *p->validityFailure < 1,
          "invalid common-event failure bound");
  if (!p->outputScale) return "missing actual output scale";
  scale(*p->outputScale);
  return p->residual ? "" : "missing primitive residual provider";
}

NoiseBounds decoded(const PrimitiveResidual& r) {
  // Revalidate at the external boundary (the public fields can be modified).
  bound(r.bounds.meanBias); bound(r.bounds.centeredSigma); bound(r.bounds.absolute);
  scale(r.unitsPerDecoded);
  return {quotient(r.bounds.meanBias, r.unitsPerDecoded), quotient(r.bounds.centeredSigma, r.unitsPerDecoded),
          quotient(r.bounds.absolute, r.unitsPerDecoded)};
}

}  // namespace

NoiseBounds::NoiseBounds(double b, double s, double a) : meanBias(b), centeredSigma(s), absolute(a) {
  bound(b); bound(s); bound(a);
}
NoiseBounds NoiseBounds::fromVarianceProxy(double b, double v, double a) {
  bound(v);
  return {b, upperSqrt(v), a};
}
double NoiseBounds::modulusFromComponentBound(double component) {
  bound(component);
  return product(upperSqrt(2), component);
}

NoiseTransferResult evaluateNoiseTransfer(const NumericDAG& dag, PlanRange x, unsigned inputLevel,
                                         double fullApproximationError, const ValidatedNoiseProfile* p) {
  bound(fullApproximationError);
  auto result = schedule(dag, x, inputLevel);
  result.fullApproximationError = fullApproximationError;
  result.reason = unavailable(p, x);
  if (!result.reason.empty()) return result;

  // Commit numeric node records only after all required sites are covered.
  std::vector<NoiseBounds> values;
  values.reserve(dag.nodes.size());
  for (std::size_t i = 0; i < dag.nodes.size(); ++i) {
    const auto& n = dag.nodes[i];
    const auto& rec = result.nodes[i];
    std::vector<NoiseBounds> args;
    for (unsigned j : n.inputs) args.push_back(values[j]);
    // The accumulator starts as the first term (not an extra add operation).
    // For PtMul touching term 0 and for a Drop of operand 0 we update it below.
    std::optional<NoiseBounds> acc;
    if (!args.empty()) acc = args.front();
    for (const auto& s : rec.sites) {
      auto residual = p->residual(s);
      if (!residual || residual->commonEvent != p->commonEvent) {
        result.reason = std::string(residual ? "mismatched common event for " : "missing residual for ") +
                        name(s.primitive) + " at node " + std::to_string(i) +
                        ", operand " + std::to_string(s.operand);
        return result;
      }
      const auto eta = decoded(*residual);
      switch (s.primitive) {
        case Kind::Source: acc = eta; break;
        case Kind::Drop:
        case Kind::PtMul: {
          auto& term = args[s.operand];
          if (s.primitive == Kind::PtMul) term = amplify(term, rec.symbolic.operandGains[s.operand]);
          term = add(term, eta);
          if (s.operand == 0) acc = term;
          break;
        }
        case Kind::Mul: acc = add(multiply(args[0], args[1], rec.symbolic), eta); break;
        case Kind::Add: acc = add(add(*acc, args[s.operand]), eta); break;
        case Kind::Relin: case Kind::Rescale: case Kind::AddPlain: acc = add(*acc, eta); break;
      }
    }
    values.push_back(acc.value());
  }
  // P(x)-activation(x) is deterministic for fixed ideal x. It shifts the
  // conditional mean/envelope once and never contributes to centered sigma.
  const auto total = amplify(add(values[dag.output], {fullApproximationError, 0, fullApproximationError}),
                             *p->outputScale);
  result.canonical = CanonicalNoiseResult{total.meanBias, total.centeredSigma,
      product(total.centeredSigma, total.centeredSigma), total.absolute, *p->validityFailure,
      p->contractId, p->commonEvent};
  for (std::size_t i = 0; i < values.size(); ++i) result.nodes[i].decoded = values[i];
  return result;
}

NoiseTransferResult evaluateNoiseTransfer(const PolyCandidate& c, const ValidatedNoiseProfile* p) {
  return evaluateNoiseTransfer(c.dag, c.idealInput, c.inputLevel, c.approximationError, p);
}

}  // namespace snabf::pass2
