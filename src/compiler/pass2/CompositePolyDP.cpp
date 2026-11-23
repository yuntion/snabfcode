#include "CompositePolyDP.h"

#include "IntervalCertifier.h"
#include "PolyTemplates.h"
#include "ReluRecipe.h"
#include "ReluBernstein.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace snabf::pass2 {
namespace {

using Id = unsigned;
using Range = PlanRange;
using std::isfinite;

void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}

bool valid(Range r) { return isfinite(r.lo) && isfinite(r.hi) && r.lo <= r.hi; }
bool same(Range a, Range b) { return a.lo == b.lo && a.hi == b.hi; }
Range range(Interval1D r) { return {r.lo, r.hi}; }
Interval1D certRange(Range r) { return {r.lo, r.hi}; }
double magnitude(Range r) { return std::max(std::abs(r.lo), std::abs(r.hi)); }

// Binary64 operations with outward endpoints. Never silently clip overflow,
// and never rely on long double being wider than double (e.g. on Windows).
static_assert(std::numeric_limits<double>::is_iec559);
double outward(double x, bool upper) {
  const double inf = std::numeric_limits<double>::infinity();
  const double y = std::nextafter(x, upper ? inf : -inf);
  if (!isfinite(x) || !isfinite(y)) throw std::overflow_error("unrepresentable Pass 2 bound");
  return y;
}

double upSum(double a, double b) {
  if (a == 0 || b == 0) return a == 0 ? b : a;
  return outward(a + b, true);
}

double upProduct(double a, double b) {
  if (a == 0 || b == 0) return 0;
  if (a == 1 || b == 1) return a == 1 ? b : a;
  return outward(a * b, true);
}

Range plus(Range a, Range b) {
  return {a.lo == 0 || b.lo == 0 ? a.lo + b.lo : outward(a.lo + b.lo, false),
          a.hi == 0 || b.hi == 0 ? a.hi + b.hi : outward(a.hi + b.hi, true)};
}

Range times(Range a, Range b) {
  if (same(a, {0, 0}) || same(b, {0, 0})) return {0, 0};
  if (same(a, {1, 1})) return b;
  if (same(b, {1, 1})) return a;
  Range result{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  for (double x : {a.lo, a.hi}) {
    for (double y : {b.lo, b.hi}) {
      const double p = x * y;
      const bool exact = x == 0 || y == 0 || std::abs(x) == 1 || std::abs(y) == 1;
      result.lo = std::min(result.lo, exact ? p : outward(p, false));
      result.hi = std::max(result.hi, exact ? p : outward(p, true));
    }
  }
  return result;
}

Range square(Range a) {
  const bool crossesZero = a.lo <= 0 && a.hi >= 0;
  const double near = crossesZero ? 0 : std::min(std::abs(a.lo), std::abs(a.hi));
  const double far = magnitude(a);
  return {std::max(0.0, times({near, near}, {near, near}).lo), upProduct(far, far)};
}

Range intersect(Range a, Range b) {
  Range result{std::max(a.lo, b.lo), std::min(a.hi, b.hi)};
  if (!valid(result)) throw std::runtime_error("inconsistent Pass 2 range enclosures");
  return result;
}

unsigned parentDepth(const NumericDAG& dag, const std::vector<Id>& inputs) {
  unsigned d = 0;
  for (Id parent : inputs) d = std::max(d, dag.nodes[parent].depth);
  return d;
}

Id append(NumericDAG& dag, std::string op, std::vector<Id> ids,
          std::vector<double> ws = {}, double c = 0) {
  const unsigned depth = parentDepth(dag, ids) + (op == "mul" || op == "linear");
  dag.output = static_cast<Id>(dag.nodes.size());
  dag.nodes.push_back({std::move(op), std::move(ids), std::move(ws), c, depth});
  return dag.output;
}

unsigned ceilSqrt(unsigned n) {
  unsigned r = 1;
  while (r * r < n) ++r;  // n <= 2 * largest supported degree.
  return r;
}

unsigned nominalDepth(unsigned degree) {
  unsigned result = 0;
  for (unsigned n = degree - 1; n; n >>= 1) ++result;
  return result;
}

unsigned nominalPS(unsigned degree) {
  const unsigned m = ceilSqrt(2 * degree);
  return m + degree / m - 2;
}

void verifyPolynomial(const StagePolynomial& p, unsigned degree) {
  require(p.degree == degree && p.coefficients.size() == degree + 1,
          "stage polynomial degree/coefficients mismatch");
  for (unsigned i = 0; i <= degree; ++i) {
    require(isfinite(p.coefficients[i]) && (i % 2 || p.coefficients[i] == 0),
            "expected a finite odd ordinary-power polynomial");
  }
}

struct AffineBlock {
  std::vector<Id> ids;
  std::vector<double> weights;
  double constant = 0;
  void term(Id id, double w) { ids.push_back(id); weights.push_back(w); }
};

// A fresh builder uses a single cache for all baby/giant powers of this SSA
// input. A square explicitly uses the same node twice, never two recomputes.
class StageBuilder {
 public:
  StageBuilder(NumericDAG& dag, Id x) : dag_(dag), x_(x), powers_{{1, x}} {}

  Id balanced(const StagePolynomial& p) {
    AffineBlock sum;
    for (unsigned i = 1; i <= p.degree; i += 2)
      if (p.coefficients[i] != 0) sum.term(power(i), p.coefficients[i]);
    return linear(std::move(sum));
  }

  Id ps(const StagePolynomial& p) {
    // p(x) = x*q(z), z=x^2. Split q into width-b baby blocks; Horner in
    // z^b. A plaintext-only top block is folded into the next linear rather
    // than inventing either a ciphertext constant or a nonscalar multiply.
    const unsigned count = (p.degree + 1) / 2;
    const unsigned b = ceilSqrt(count);
    std::optional<Id> acc;
    double constant = 0;
    for (unsigned block = (count + b - 1) / b; block-- > 0;) {
      AffineBlock baby;
      baby.constant = p.coefficients[2 * block * b + 1];
      for (unsigned j = 1; j < b && block * b + j < count; ++j) {
        const double c = p.coefficients[2 * (block * b + j) + 1];
        if (c != 0) baby.term(zPower(j), c);
      }
      if (acc) {
        Id giantProduct = multiply(*acc, zPower(b));
        if (baby.ids.empty()) {
          acc = baby.constant == 0 ? giantProduct
                                   : append(dag_, "add", {giantProduct}, {}, baby.constant);
        } else {
          Id local = linear(std::move(baby));
          acc = append(dag_, "add", {giantProduct, local});
        }
      } else {
        if (constant != 0) baby.term(zPower(b), constant);
        if (!baby.ids.empty()) acc = linear(std::move(baby));
        else constant = baby.constant;
      }
    }
    if (acc) return multiply(x_, *acc);
    return append(dag_, "linear", {x_}, {constant});
  }

 private:
  Id multiply(Id a, Id b) { return append(dag_, "mul", {a, b}); }
  Id power(unsigned k) {
    if (auto it = powers_.find(k); it != powers_.end()) return it->second;
    const Id a = power(k / 2);
    const Id b = power(k - k / 2);
    return powers_.emplace(k, multiply(a, b)).first->second;
  }
  Id zPower(unsigned k) {
    if (k == 1) return power(2);
    if (auto it = powers_.find(2*k); it != powers_.end()) return it->second;
    const Id a = zPower(k/2), b = zPower(k-k/2);
    return powers_.emplace(2*k, multiply(a, b)).first->second;
  }
  Id linear(AffineBlock block) {
    if (block.ids.empty()) block.term(x_, 0);  // Constant in ciphertext form.
    return append(dag_, "linear", std::move(block.ids), std::move(block.weights), block.constant);
  }
  NumericDAG& dag_;
  Id x_;
  std::map<unsigned, Id> powers_;
};

}  // namespace

DAGCost costDAG(const NumericDAG& dag) {
  require(!dag.nodes.empty() && dag.output < dag.nodes.size(), "missing numeric DAG output");
  DAGCost c;
  std::vector<bool> used(dag.nodes.size(), false);
  used[dag.output] = true;
  for (std::size_t i = 0; i < dag.nodes.size(); ++i) {
    const PlanNode& n = dag.nodes[i];
    require(isfinite(n.constant), "nonfinite DAG constant");
    for (Id parent : n.inputs) require(parent < i, "non-topological DAG input ID");
    const unsigned d = parentDepth(dag, n.inputs);
    const unsigned end = d + (n.op == "linear" || n.op == "mul");
    require(n.depth == end, "incorrect scheduled DAG depth");
    if (c.byDepth.size() <= end) c.byDepth.resize(end + 1);
    auto& work = c.byDepth[d];
    require((i == 0) == (n.op == "input"), "DAG must have exactly one initial input");
    if (n.op == "input") {
      require(n.inputs.empty() && n.weights.empty() && n.constant == 0, "malformed input");
    } else if (n.op == "mul") {
      require(n.inputs.size() == 2 && n.weights.empty() && n.constant == 0, "malformed mul");
      ++c.ctMults; ++c.relins; ++c.rescales;
      ++(n.inputs[0] == n.inputs[1] ? work.ctSquares : work.ctProducts);
      ++work.relins; ++work.rescales;
    } else if (n.op == "linear" || n.op == "add") {
      const bool linear = n.op == "linear";
      require(!n.inputs.empty() && (linear ? n.weights.size() == n.inputs.size() : n.weights.empty()),
              "malformed linear/add");
      for (double w : n.weights) require(isfinite(w), "nonfinite DAG weight");
      c.ptMults += static_cast<unsigned>(n.weights.size());
      c.additions += static_cast<unsigned>(n.inputs.size() - 1) + (n.constant != 0);
      c.rescales += linear;
      work.ptMults += static_cast<unsigned>(n.weights.size());
      work.ctAdds += static_cast<unsigned>(n.inputs.size() - 1);
      work.rescales += linear;
      c.byDepth[end].ptAdds += (n.constant != 0);  // After a linear's shared rescale.
    } else {
      throw std::invalid_argument("unknown numeric DAG op: " + n.op);
    }
    for (Id parent : n.inputs) {
      const unsigned from = dag.nodes[parent].depth, span = d-from;
      if (!span) continue;
      c.alignmentDrops += span;
      auto& reductions = c.byDepth[from].levelReductions;
      if (reductions.size() <= span) reductions.resize(span+1);
      ++reductions[span];
      for (unsigned k = from; k < d; ++k) ++c.byDepth[k].alignmentDrops;
    }
  }
  for (std::size_t i = dag.nodes.size(); i-- > 0;) {
    require(used[i], "dead numeric DAG node");
    for (Id parent : dag.nodes[i].inputs) used[parent] = true;
  }
  c.depth = dag.nodes[dag.output].depth;
  return c;
}

double coreToActivationError(PlanRange input, double coreError, double templateRefError) {
  require(valid(input) && isfinite(coreError) && coreError >= 0 &&
              isfinite(templateRefError) && templateRefError >= 0, "invalid reconstruction error");
  return upProduct(magnitude(input), upSum(coreError, templateRefError));
}

double evaluateDAG(const NumericDAG& dag, double x) {
  (void)costDAG(dag);
  require(isfinite(x), "nonfinite DAG input");
  std::vector<double> values;
  values.reserve(dag.nodes.size());
  for (const PlanNode& n : dag.nodes) {
    double v = n.constant;
    if (n.op == "input") v = x;
    else if (n.op == "mul") v = values[n.inputs[0]] * values[n.inputs[1]];
    else if (n.op == "add") { for (Id id : n.inputs) v += values[id]; }
    else for (std::size_t i = 0; i < n.inputs.size(); ++i)
      v = std::fma(n.weights[i], values[n.inputs[i]], v);
    values.push_back(v);
  }
  return values[dag.output];
}

namespace {

struct Evidence { StagePolynomial p; StageCertificate cert; };

// Fit on I, certify on (I,A). DAG schedules for the same transition need
// neither another fit nor another interval proof. Keep caches per search.
class StageCache {
 public:
  const Evidence& get(StageFunction fn, unsigned d, Range I, Range A) {
    const FitKey fk{fn, d, I.lo, I.hi};
    const CertKey ck{fk, A.lo, A.hi};
    if (auto it = certs_.find(ck); it != certs_.end()) return it->second;
    auto fit = fits_.find(fk);
    if (fit == fits_.end()) {
      StagePolynomial p = generateStage(fn, d, certRange(I));
      verifyPolynomial(p, d);
      fit = fits_.emplace(fk, std::move(p)).first;
    }
    const StageCertificate c = certifyStage(fn, fit->second, certRange(I), certRange(A));
    require(same(range(c.Iin), I) && same(range(c.Ain), A) &&
                same(range(c.J), {std::min(I.lo, A.lo), std::max(I.hi, A.hi)}) &&
                valid(range(c.Iout)) && valid(range(c.Aout)) &&
                isfinite(c.epsilon) && c.epsilon >= 0 && isfinite(c.L) && c.L >= 0,
            "invalid stage certificate");
    return certs_.emplace(ck, Evidence{fit->second, c}).first->second;
  }
 private:
  using FitKey = std::tuple<StageFunction, unsigned, double, double>;
  using CertKey = std::tuple<FitKey, double, double>;
  std::map<FitKey, StagePolynomial> fits_;
  std::map<CertKey, Evidence> certs_;
};

struct Prefix {
  PolyCandidate p;
  Range ideal, actual;
  double e = 0, eApprox = 0, gain = 1;
};

Prefix extend(const Prefix& prior, const Evidence& evidence, StageFunction fn, bool ps) {
  Prefix next = prior;
  auto& p = next.p;
  const auto& [poly, c] = evidence;
  const Id input = p.dag.output;
  StageBuilder builder(p.dag, input);
  const Id output = ps ? builder.ps(poly) : builder.balanced(poly);
  const unsigned depth = nominalDepth(poly.degree), mults = nominalPS(poly.degree);
  next.e = upSum(upProduct(c.L, prior.e), c.epsilon);
  next.eApprox = upSum(upProduct(c.L, prior.eApprox), c.epsilon);
  next.gain = upProduct(prior.gain, c.L);
  next.ideal = range(c.Iout); next.actual = range(c.Aout);
  const std::string schedule = ps ? "ps" : "balanced";
  p.stages.push_back({std::string(stageName(fn)), schedule, poly.degree, input, output,
                      depth, mults, p.dag.nodes[output].depth - p.dag.nodes[input].depth,
                      poly.coefficients, range(c.Iin), range(c.Ain), range(c.J),
                      next.ideal, next.actual, c.epsilon, c.L, prior.e, next.e, next.gain});
  p.nominalDepth += depth;
  p.nominalPSMults += mults;
  p.cost = costDAG(p.dag);
  p.id += ".d" + std::to_string(poly.degree) + "." + schedule;
  return next;
}

// All comparisons are componentwise; equal vectors with different numerical
// schedules are NOT strict dominance. A tested candidate can be outside the
// deterministic frontier yet have an unknown (not certified worse) CKKS noise.
struct Comparison {
  bool weak = true, strict = false;
  template<class T> void lower(T a, T b) { weak &= a <= b; strict |= a < b; }
  void subset(Range a, Range b) { lower(b.lo, a.lo); lower(a.hi, b.hi); }
  void cost(const DAGCost& a, const DAGCost& b) {
    if (a.byDepth.empty() || b.byDepth.empty()) { weak = false; return; }
    lower(a.depth, b.depth);
    // Nonnegative independent prices by primitive and absolute active level:
    // equal inputLevel is checked by the callers. No monotone-price assumption.
    const DepthWork zero{};
    for (std::size_t i = 0, n = std::max(a.byDepth.size(), b.byDepth.size()); i < n; ++i) {
      const auto& x = i < a.byDepth.size() ? a.byDepth[i] : zero;
      const auto& y = i < b.byDepth.size() ? b.byDepth[i] : zero;
      lower(x.ctProducts,y.ctProducts); lower(x.ctSquares,y.ctSquares); lower(x.relins,y.relins);
      lower(x.ptMults,y.ptMults); lower(x.rescales,y.rescales); lower(x.ctAdds,y.ctAdds);
      lower(x.ptAdds,y.ptAdds); lower(x.alignmentDrops,y.alignmentDrops);
      for (std::size_t k = 1; k < std::max(x.levelReductions.size(), y.levelReductions.size()); ++k)
        lower(k < x.levelReductions.size() ? x.levelReductions[k] : 0u,
              k < y.levelReductions.size() ? y.levelReductions[k] : 0u);
    }
  }
};

template<class T, class Dominates>
void insertFront(std::vector<T>& front, T value, Dominates dominates) {
  if (std::any_of(front.begin(), front.end(), [&](const T& x) { return dominates(x, value); })) return;
  std::erase_if(front, [&](const T& x) { return dominates(value, x); });
  front.push_back(std::move(value));
}

NoiseContract noiseContract(const PolyCandidate& p) {
  NoiseContract noise;
  for (std::size_t i = 0; i < p.dag.nodes.size(); ++i) {
    const PlanNode& n = p.dag.nodes[i];
    NodeTransfer t;
    if (n.op == "input") {
      t.range = p.actualInput;
    } else if (n.op == "mul") {
      const auto a = noise.nodes[n.inputs[0]].range, b = noise.nodes[n.inputs[1]].range;
      t.range = n.inputs[0] == n.inputs[1] ? square(a) : times(a, b);
      t.operandGains = {magnitude(b), magnitude(a)};
      t.productRemainder = true;
    } else {
      t.range = {n.constant, n.constant};
      for (std::size_t j = 0; j < n.inputs.size(); ++j) {
        const double w = n.op == "add" ? 1 : n.weights[j];
        t.range = plus(t.range, times({w, w}, noise.nodes[n.inputs[j]].range));
        t.operandGains.push_back(std::abs(w));
      }
    }
    // Tighten the same exact-real value with the independent stage proof.
    for (const StageTrace& s : p.stages)
      if (s.outputNode == i) t.range = intersect(t.range, s.actualOut);
    if (i == p.dag.output) t.range = intersect(t.range, p.actualOutput);
    noise.nodes.push_back(std::move(t));
  }
  noise.nodes[p.dag.output].outputSensitivity = 1;
  for (std::size_t i = p.dag.nodes.size(); i-- > 0;) {
    const auto& n = p.dag.nodes[i];
    const auto& t = noise.nodes[i];
    for (std::size_t j = 0; j < n.inputs.size(); ++j) {
      auto& g = noise.nodes[n.inputs[j]].outputSensitivity;
      g = upSum(g, upProduct(t.outputSensitivity, t.operandGains[j]));
    }
  }
  return noise;
}

PolyCandidate reconstruct(Prefix prefix, const PolyTemplateSpec& spec) {
  PolyCandidate p = std::move(prefix.p);
  const double offset = spec.reconstructionOffset;
  const Range constant{offset, offset};
  const Id core = p.dag.output;
  const Id halfX = append(p.dag, "linear", {0}, {offset});
  const Id product = append(p.dag, "mul", {0, core});
  append(p.dag, "add", {halfX, product});  // x/2 + x*core, not sigmoid alone.
  p.cost = costDAG(p.dag);
  require(p.cost.depth <= p.inputLevel, "internal reconstruction level underflow");
  p.outputLevel = p.inputLevel - p.cost.depth;
  p.coreGain = prefix.gain;
  p.bypassGain = magnitude(plus(constant, prefix.actual));
  p.outputGain = upSum(p.bypassGain, upProduct(magnitude(p.actualInput), p.coreGain));
  p.approximationError = coreToActivationError(p.idealInput, prefix.eApprox, spec.templateRefError);
  // x'*(c+P(x')) - x*(c+g(x)) = (x'-x)*(c+P(x')) + x*(P(x')-g(x)).
  p.outputError = upSum(upProduct(p.inputError, p.bypassGain),
                        coreToActivationError(p.idealInput, prefix.e, spec.templateRefError));
  const Range referenceCore = plus(prefix.ideal, {-spec.templateRefError, spec.templateRefError});
  p.idealOutput = times(p.idealInput, plus(constant, referenceCore));
  p.actualOutput = intersect(times(p.actualInput, plus(constant, prefix.actual)),
                             plus(p.idealOutput, {-p.outputError, p.outputError}));
  p.noise = noiseContract(p);
  return p;
}

}  // namespace

DPResult synthesizeNativeRelu(Range input,const DPOptions &o) {
  require(valid(input) && input.lo<=0 && input.hi>=0 && o.inputError==0 &&
          isfinite(o.maxError) && o.maxError>0,"invalid native ReLU domain/error contract");
  DPResult r;
  if(o.minOutputLevel>o.inputLevel || !o.composed) return r;
  const auto chain=reluRecipe(certRange(input));
  for(auto &s:chain) if(std::find(o.degrees.begin(),o.degrees.end(),s.degree)==o.degrees.end()) return r;
  const auto proof=certifyReluReconstruction(chain,certRange(input));
  if(proof.epsilon>o.maxError) {++r.errorRejected;return r;}
  int exponent=0;
  const double frac=std::frexp(proof.epsilon,&exponent);
  const double coverBudget=frac==0 ? o.maxError : std::min(o.maxError,std::ldexp(1.,exponent-(frac==0.5)));
  const auto independent=relu_bernstein::prove(chain,certRange(input),coverBudget);
  const auto limit=std::min(o.maxDepth,o.inputLevel-o.minOutputLevel);
  for(bool ps:{false,true}) {
    if(ps ? !o.ps : !o.balanced) continue;
    Prefix s; s.ideal=s.actual=s.p.idealInput=s.p.actualInput=input;
    s.p.inputLevel=o.inputLevel; s.p.activation="relu"; s.p.templateName=s.p.id="relu_composite";
    append(s.p.dag,"input",{});
    for(const auto &poly:chain) {
      auto fn=StageFunction::ReluSignPolynomial;
      auto cert=certifyStage(fn,poly,certRange(s.ideal),certRange(s.actual));
      s=extend(s,{poly,cert},fn,ps); ++r.transitions;
    }
    auto &p=s.p; const Id core=p.dag.output;
    const Id half=append(p.dag,"linear",{0},{0.5});
    const Id product=append(p.dag,"mul",{half,core});
    append(p.dag,"add",{half,product});
    p.cost=costDAG(p.dag);
    if(p.cost.depth>limit){++r.depthRejected;continue;}
    p.outputLevel=p.inputLevel-p.cost.depth;
    p.relu=PolyCandidate::ReluProof{{proof.leftEpsilon,proof.rightEpsilon},
      {independent.upper[0].get_str(),independent.upper[1].get_str()},independent.cells,coverBudget};
    p.approximationError=p.outputError=proof.epsilon;
    const auto factor=times({0.5,0.5},plus({1,1},s.actual));
    p.coreGain=upProduct(0.5,s.gain); p.bypassGain=magnitude(factor);
    p.outputGain=upSum(p.bypassGain,upProduct(magnitude(input),p.coreGain));
    p.idealOutput={std::max(0.,input.lo),std::max(0.,input.hi)};
    p.actualOutput=intersect(times(input,factor),plus(p.idealOutput,{-p.outputError,p.outputError}));
    p.noise=noiseContract(p); ++r.feasible;
    if(o.keepAllFeasible)r.allFeasible.push_back(p);
    insertFront(r.frontier,std::move(p),dominates);
  }
  return r;
}

bool dominates(const PolyCandidate& a, const PolyCandidate& b) {
  if (a.activation != b.activation || a.inputLevel != b.inputLevel || a.inputError != b.inputError ||
      !same(a.idealInput, b.idealInput) || !same(a.actualInput, b.actualInput)) return false;
  Comparison c;
  c.cost(a.cost, b.cost); c.lower(b.outputLevel, a.outputLevel);
  c.lower(a.approximationError, b.approximationError); c.lower(a.outputError, b.outputError);
  c.lower(a.outputGain, b.outputGain); c.subset(a.idealOutput, b.idealOutput);
  c.subset(a.actualOutput, b.actualOutput);
  return c.weak && c.strict;
}

DPResult synthesizePolynomials(std::string_view activation, Range input, const DPOptions& options) {
  require(valid(input), "invalid ideal input range");
  require(isfinite(options.inputError) && options.inputError >= 0, "invalid decoded input error");
  require(options.maxError >= 0, "invalid error budget");  // +infinity disables this constraint.
  auto degrees = options.degrees;
  std::sort(degrees.begin(), degrees.end());
  degrees.erase(std::unique(degrees.begin(), degrees.end()), degrees.end());
  for (unsigned d : degrees)
    require(std::find(kCandidateDegrees.begin(), kCandidateDegrees.end(), d) != kCandidateDegrees.end(),
            "unsupported Pass 2 candidate degree");
  const auto& templates = templatesFor(activation);
  require(!templates.empty(), "unknown Pass 2 activation");
  DPResult result;
  if (options.minOutputLevel > options.inputLevel) return result;
  const unsigned limit = std::min(options.maxDepth, options.inputLevel - options.minOutputLevel);
  const Range actualInput = plus(input, {-options.inputError, options.inputError});
  StageCache cache;
  for (const auto& spec : templates) {
    require(!spec.stages.empty() && isfinite(spec.reconstructionOffset) && isfinite(spec.templateRefError) &&
                spec.templateRefError >= 0 && spec.reconstruction == Reconstruction::MultiplyOriginalInputByOffsetPlusCore,
            "invalid reference template");
    if (spec.stages.size() == 1 ? !options.direct : !options.composed) continue;
    Prefix seed;
    seed.ideal = seed.p.idealInput = input; seed.actual = seed.p.actualInput = actualInput;
    seed.e = seed.p.inputError = options.inputError;
    seed.p.activation = std::string(activation); seed.p.id = seed.p.templateName = spec.id;
    seed.p.templateRefError = spec.templateRefError; seed.p.inputLevel = options.inputLevel;
    append(seed.p.dag, "input", {});
    std::vector<Prefix> layer{std::move(seed)};
    for (std::size_t j = 0; j < spec.stages.size() && !layer.empty(); ++j) {
      const bool last = j + 1 == spec.stages.size();
      std::vector<Prefix> nextLayer;
      for (const auto& state : layer) for (unsigned degree : degrees) {
        const Evidence* ev;
        try { ev = &cache.get(spec.stages[j], degree, state.ideal, state.actual); }
        catch (const std::bad_alloc&) { throw; }
        catch (const std::exception& e) {
          result.certificationFailures.push_back(state.p.id + ".d" + std::to_string(degree) + ": " + e.what());
          continue;
        }
        for (bool ps : {false, true}) {
          if (ps ? !options.ps : !options.balanced) continue;
          ++result.transitions;
          try {
            auto next = extend(state, *ev, spec.stages[j], ps);
            // The still-live original x and final ciphertext product require
            // one more level even if the last fitted coefficient is tiny.
            if (next.p.cost.depth >= limit) { ++result.depthRejected; continue; }
            if (!last) {
              // At most 12 first-stage schedules for the registered templates.
              // A later stage sees the actual range, gain, and entire DAG.
              // Keep every feasible prefix and compare only completed DAGs.
              nextLayer.push_back(std::move(next));
            } else {
              auto candidate = reconstruct(std::move(next), spec);
              if (candidate.outputError > options.maxError) { ++result.errorRejected; continue; }
              ++result.feasible;
              if (options.keepAllFeasible) result.allFeasible.push_back(candidate);
              insertFront(result.frontier, std::move(candidate), dominates);
            }
          } catch (const std::overflow_error& e) {
            result.certificationFailures.push_back(state.p.id + ".d" + std::to_string(degree) + ": " + e.what());
          }
        }
      }
      layer = std::move(nextLayer);
    }
  }
  result.dominated = result.feasible - result.frontier.size();
  std::sort(result.frontier.begin(), result.frontier.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  return result;
}

DPResult synthesizeSiLU(Range input, const DPOptions& options) {
  return synthesizePolynomials("silu", input, options);
}

}  // namespace snabf::pass2
