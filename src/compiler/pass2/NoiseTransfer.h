#pragma once

#include "CompositePolyDP.h"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace snabf::pass2 {

// For EACH fixed real ideal input and slot, under the law conditional on G:
// |E[Re e]|, |E[Im e]| <= meanBias; each centered component has MGF at most
// exp(t^2 * centeredSigma^2 / 2) for all real t; |e| <= absolute a.s.
// absolute is a COMPLEX MODULUS envelope, not a component bound or quantile.
struct NoiseBounds {
  double meanBias, centeredSigma, absolute;
  NoiseBounds(double meanBias, double centeredSigma, double absolute);
  // A centered sub-Gaussian variance PROXY, not a measured second moment.
  static NoiseBounds fromVarianceProxy(double meanBias, double proxy, double absolute);
  static double modulusFromComponentBound(double realAndImagBound);
};

enum class NoisePrimitive { Source, Drop, Mul, Relin, PtMul, Add, Rescale, AddPlain };

struct NoiseSite {
  std::size_t node;
  NoisePrimitive primitive;
  // Operand position for Drop/PtMul; right-hand term position for Add; else 0.
  std::size_t operand;
  // A multi-prime level_reduce is ONE site covering the entire level span.
  unsigned inputLevel, outputLevel;
};

struct PrimitiveResidual {
  std::string commonEvent;
  NoiseBounds bounds;
  // 1 for decoded residuals; actual local physical Delta for canonical ones.
  // A fixed conversion factor, NOT a random scale or a nominal log2 scale.
  double unitsPerDecoded;
};

// Trust boundary: the caller binds this profile to the exact materialized
// schedule and attests that each returned residual is validated for its site
// (parameters, scales, keys, plaintext encodings and reachable noisy operands),
// on the SAME positive-probability event G, uniformly for x in idealDomain.
// G must cover domain, no-wrap and primitive validity across the execution
// and all claimed slots. It is insufficient to validate only real, noiseless
// operands: the reachable noisy operands can leave the interval and be complex.
// Unconditional MGFs or MGFs on separate local events cannot be relabeled as
// conditional MGFs on their intersection. Event/provenance IDs are audit tags,
// not evidence. No statement is made about mixing different ideal plaintexts.
struct ValidatedNoiseProfile {
  std::string contractId, commonEvent;
  std::optional<PlanRange> idealDomain;
  std::optional<double> validityFailure;  // Upper bound on Pr[G^c], charged once.
  std::optional<double> outputScale;      // Actual, fixed physical output Delta.
  // Source is the COMPLETE incoming error relative to the ideal activation
  // input (including upstream error). Other sites supply only the incremental
  // decoded-output residual relative to the operation on its noisy parents.
  // PtMul/AddPlain include encoding/physical-scale effects; Rescale includes
  // any deviation from decoded identity. No primitive defaults to exact.
  std::function<std::optional<PrimitiveResidual>(const NoiseSite&)> residual;
};

struct NoiseNodeResult {
  NodeTransfer symbolic;                 // For the ideal real arithmetic DAG.
  std::vector<NoiseSite> sites;          // Actual primitive order and counts.
  std::optional<NoiseBounds> decoded;    // Evaluation vs exact polynomial only.
};

struct CanonicalNoiseResult {
  // Per-real/imag-component bias and MGF proxy; absolute bounds slot modulus.
  double meanBias, centeredSigma, varianceProxy, absolute, validityFailure;
  std::string contractId, commonEvent;
};

struct NoiseTransferResult {
  DAGCost cost{};  // alignmentDrops counts primes; sites count actual drop calls.
  double fullApproximationError = 0;  // Decoded deterministic bound, also when symbolic.
  std::vector<NoiseNodeResult> nodes;
  std::optional<CanonicalNoiseResult> canonical;  // Engaged ONLY if conditional.
  std::string reason;                             // Why the result is symbolic.
};

// The materialized DAG must use real weights/constants and real ideal inputs;
// noisy CKKS slots can be complex. A shared-rescale linear/weighted_sum means
// one PtMul per operand, a left fold of Add, one Rescale, then AddPlain iff
// the constant is nonzero. Contracts do not transfer to a different lowering.
// fullApproximationError must independently
// bound |P(x)-activation(x)| on idealInput, *without* incoming/CKKS error.
// It is charged once, at the output. A missing profile/site/domain/scale yields
// a symbolic result. Malformed DAGs/contracts and numerical overflow throw.
NoiseTransferResult evaluateNoiseTransfer(const NumericDAG& dag, PlanRange idealInput,
                                         unsigned inputLevel, double fullApproximationError,
                                         const ValidatedNoiseProfile* profile = nullptr);
NoiseTransferResult evaluateNoiseTransfer(const PolyCandidate& candidate,
                                         const ValidatedNoiseProfile* profile = nullptr);

}  // namespace snabf::pass2
