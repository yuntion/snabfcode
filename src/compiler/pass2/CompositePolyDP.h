#ifndef SNABF_PASS2_COMPOSITE_POLY_DP_H
#define SNABF_PASS2_COMPOSITE_POLY_DP_H

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace snabf::pass2 {

// Stable backend interface. IDs are topological vector indices; node 0 is x.
// Align COPIES of all inputs to level inputLevel - max(parent depths).
// input:  no operands. mul: two ciphertexts, multiply/relinearize/rescale.
// linear: constant + sum(weights[i] * inputs[i]); encode at plaintext
//         precision, accumulate compatible products, then ONE shared rescale;
//         add the optional plaintext constant at the resulting scale/level.
// add:    constant + sum(inputs); same-scale addition, no rescale.
// mul and linear cost one active Q level each; add costs zero. Non-linear
// nodes have empty weights; only linear and add may carry a constant.
struct PlanNode {
  std::string op;
  std::vector<unsigned> inputs;
  std::vector<double> weights;
  double constant = 0;
  unsigned depth = 0;  // Consumed levels, not number of remaining limbs.
  bool operator==(const PlanNode&) const = default;
};

struct NumericDAG {
  std::vector<PlanNode> nodes;
  unsigned output = 0;
  bool operator==(const NumericDAG&) const = default;
};

struct DepthWork {
  unsigned ctProducts = 0, ctSquares = 0, relins = 0, ptMults = 0, rescales = 0;
  unsigned ctAdds = 0, ptAdds = 0, alignmentDrops = 0;
  // Number of level-reduce operations starting here and dropping exactly k
  // primes, indexed by k. One multi-prime reduce is not k native operations.
  std::vector<unsigned> levelReductions;
};

struct DAGCost {
  unsigned depth = 0, ctMults = 0, relins = 0, ptMults = 0, rescales = 0;
  unsigned additions = 0, alignmentDrops = 0;  // Prime drops per operand use.
  // Index d = consumed depth BEFORE the primitive, or active level inputLevel-d.
  // A level's plaintext encoding count is ptMults + ptAdds. Missing bins are zero.
  std::vector<DepthWork> byDepth;
};

// Both validate every operand, arity, scheduled depth and output reachability.
DAGCost costDAG(const NumericDAG& dag);
double evaluateDAG(const NumericDAG& dag, double x);

struct PlanRange { double lo = 0, hi = 0; };

// Full activation residual of x*(offset+core) at the SAME x. Both error
// arguments are bounds in core units, including the template/reference term.
double coreToActivationError(PlanRange idealInput, double coreError, double templateRefError);

struct StageTrace {
  std::string stage, schedule;  // "ps" or "balanced"
  unsigned degree = 0, inputNode = 0, outputNode = 0;
  unsigned nominalDepth = 0, nominalPSMults = 0, actualDepth = 0;
  std::vector<double> coefficients;  // Ordinary powers of the stage input.
  PlanRange idealIn, actualIn, hullIn, idealOut, actualOut;
  double residual = 0, lipschitz = 0, errorIn = 0, errorOut = 0;
  double cumulativeGain = 0;
};

struct NodeTransfer {
  PlanRange range;  // Real DAG value enclosure, before CKKS roundoff.
  std::vector<double> operandGains;  // In operand order; not independent RVs.
  bool productRemainder = false;
  // Bound on the first derivative of the full output w.r.t. an additive
  // perturbation here. Higher-order terms still use the recurrence below.
  double outputSensitivity = 0;
};

// Symbolic decoded perturbation transfer at node v:
//   E_v <= sum_i operandGains[i] * E_i + [mul] E_left * E_right + eta_v.
// eta_v is an unresolved local arithmetic error (including alignment,
// encoding, rescale and key switching as applicable). At input E_0 is supplied
// by the caller. No stochastic independence, scale or tail bound is assumed.
struct NoiseContract {
  std::vector<NodeTransfer> nodes;  // Same indices as NumericDAG.
  // Canonical bias needs (Delta_out/Delta_in)*outputGain*b_in +
  // Delta_out*approximationError + b_eval. Neither scales nor the DAG's
  // arithmetic residual/tail/validity certificates are supplied by Pass 2.
  std::optional<double> canonicalBias, canonicalVarianceProxy, validityFailure;
};

struct PolyCandidate {
  struct ReluProof {
    std::array<double,2> mpfi;
    std::array<std::string,2> bernstein;
    std::array<unsigned,2> cells;
    double bernsteinBudget;
  };
  std::optional<ReluProof> relu;
  std::string id, templateName, activation;
  std::vector<StageTrace> stages;
  NumericDAG dag;
  DAGCost cost;
  unsigned inputLevel = 0, outputLevel = 0;
  unsigned nominalDepth = 0, nominalPSMults = 0;  // Stage-only paper estimates.
  PlanRange idealInput, actualInput, idealOutput, actualOutput;
  double inputError = 0, templateRefError = 0, coreGain = 0, bypassGain = 0;  // Ref error: core units.
  double approximationError = 0;  // Full real polynomial vs reference; e0 = 0.
  double outputGain = 0;          // Includes the x bypass and terminal mul.
  double outputError = 0;         // Also includes supplied decoded input error.
  NoiseContract noise;            // Excludes all unvalidated CKKS certificates.
};

struct DPOptions {
  unsigned inputLevel = 16, maxDepth = 16, minOutputLevel = 0;
  double maxError = std::numeric_limits<double>::infinity(), inputError = 0;
  std::vector<unsigned> degrees{3, 5, 7, 9, 13, 15};
  bool direct = true, composed = true, balanced = true, ps = true;
  // Also return all completed feasible DAGs for audit. The search always
  // retains every level-feasible prefix, whether this is true or false.
  bool keepAllFeasible = false;
};

struct DPResult {
  std::vector<PolyCandidate> frontier;
  std::vector<PolyCandidate> allFeasible;  // Populated only on request.
  std::size_t transitions = 0, depthRejected = 0, errorRejected = 0;
  std::size_t prefixDominated = 0, feasible = 0, dominated = 0;  // prefixDominated always 0.
  // IDs/reasons for fits or bounds that could not be certified (fail closed).
  std::vector<std::string> certificationFailures;
};

// Searches both SiLU reference templates stage by stage. Infeasible budgets
// return an empty frontier. Malformed input throws std::invalid_argument.
DPResult synthesizeSiLU(PlanRange input, const DPOptions& options = {});
// Same search for any activation registered by PolyTemplates. Unknown names
// are rejected; a single-stage reference uses the 'direct' option.
DPResult synthesizePolynomials(std::string_view activation, PlanRange input,
                               const DPOptions& options = {});

// Native only. This must not add ReLU to the legacy template registry.
DPResult synthesizeNativeRelu(PlanRange input,const DPOptions &options);

// Strict Pareto dominance over the deterministic interface and each primitive
// at each active level (including level reductions by source and span);
// it makes no claim about as-yet uninstantiated cryptographic noise or runtime.
bool dominates(const PolyCandidate& a, const PolyCandidate& b);

}  // namespace snabf::pass2

#endif
