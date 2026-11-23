#pragma once

#include "ExactScaleTracker.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace snabf::native {

// Live, homogeneous SSA value from the native producer, not a desired scale.
// The caller checks logical shape/layout and accounts for non-scale CKKS noise.
struct NativeScaleValue {
  std::string id;
  ScaleDomain domain;
  int level = -1, slots = 0;
  ScaleState scale;
};

struct NativeScaleUse {
  std::string consumer;
  int operand_index;
  bool operator==(const NativeScaleUse &) const = default;
};

// Actual defining chain: pt = encode(call); mul = MulPlain(input, pt);
// operand.value = Rescale(mul). call_id names pt; site_id names mul. Populate
// uses from the complete SSA use lists, including sinks. No multi-use rewrite.
struct NativeScalePublicMul {
  NativeScaleValue input;
  EncodingCallContext call;
  ExactRational requested;
  std::vector<double> logical_payload;
  EncodingCallReceipt original_encoding;
  std::vector<uint64_t> dropped_q;
  std::vector<NativeScaleUse> plain_uses, product_uses, rescale_uses;
};

// Fresh SSA names reserved by the emitter. Encoding precision/branch must
// describe the real encoder; neither the scale nor the payload is caller-set.
struct NativeScaleAlignSite {
  EncodingCallContext call;
  std::string rescale_id, peer_drop_id;
};

struct NativeScaleOperand {
  NativeScaleValue value;
  std::optional<ExactRational> abs_bound; // decoded logical magnitude
  std::optional<NativeScalePublicMul> public_mul;
  std::optional<NativeScaleAlignSite> align_one;
  // Strict re-solve replay currently supports a fresh input, optionally
  // followed by the above public mul/rescale. This is that input's actual
  // dynamic encode receipt. Other producer graphs cannot authorize Tier 2.
  std::optional<EncodingCallReceipt> source_encoding;
};

struct NativeScaleAddRequest {
  std::string result_id;
  NativeScaleOperand lhs, rhs;
  std::optional<ExactRational> decoded_bias_budget; // local allocation; absent = 0
  int min_output_level = 0;
};

enum class NativeScaleMode { None, FoldIntoPlain, ExactPrimeMatch, BoundedMismatchBias, ExplicitAlignMul };

struct NativeEncodePlain {
  EncodingCallReceipt encoding; // result = call_id; tile payload to encoding.slots
  std::vector<double> payload;
};
struct NativeMulPlain {
  std::string result, ciphertext, plaintext;
  ScaleState after;
};
struct NativeRescaleOne {
  std::string result, input;
  RescaleStepReceipt step;
};
struct NativeDropLevel {
  std::string result, input;
  int from_level, to_level;
  ScaleState unchanged;
};
using NativeScalePreparation = std::variant<NativeEncodePlain, NativeMulPlain,
                                            NativeRescaleOne, NativeDropLevel>;

struct NativeAddCiphertexts {
  std::string result, lhs, rhs;
  int level;
  ScaleState lhs_before, rhs_before; // actual add inputs after all preparation
  ExactRational metadataRatio() const { return lhs_before.metadata / rhs_before.metadata; }
  ExactRational effectiveRatio() const { return lhs_before.effective / rhs_before.effective; }
  // Emit the ciphertext Add once. Its built-in integer multipliers are
  // described here, not additional instructions. A mixed gain stays absent.
  AddSubScaleAnalysis scales;
};

struct NativeScaleAddPlan {
  NativeScaleMode mode;
  ScaleDomain domain;
  int slots;
  // FoldIntoPlain replaces the three named existing definitions in place.
  // All other preparations create fresh values. The add always executes last.
  std::vector<NativeScalePreparation> preparation;
  NativeAddCiphertexts add;
  std::optional<ZeroDepthFoldResult> fold;
  ExactRational charged_decoded_bias;
  // Incremental cost over the original graph, not the cost of an existing fold site.
  int mul_depth_delta = 0, levels_consumed = 0;
  std::vector<uint64_t> dropped_q;
};

// A whole-program strict parameter re-solve, not a test on the current
// level. Return a concrete candidate to rebuild, or nullopt if unavailable.
// The callback cannot hand back replacement ciphertext scales. An invalid
// claimed solution is rejected. With no replayable roots this tier is skipped.
using NativeStrictScaleSolver =
    std::function<std::optional<ScaleDomain>(const NativeScaleAddRequest &)>;
class NativeStrictScaleRestart;
using NativeScaleAddResult = std::variant<NativeScaleAddPlan, NativeStrictScaleRestart>;

// Throws on invalid bindings, unsupported modes, or exhausted alternatives.
// No edit is applied to the input graph, including on refusal or restart.
NativeScaleAddResult lowerNativeScaleAdd(const NativeScaleAddRequest &,
                                         const NativeStrictScaleSolver &strict_solver = {});

class NativeStrictScaleRestart {
public:
  const ScaleDomain &requiredDomain() const { return domain_; }
  // Discard the old parameter plan. Pass the live join from the complete
  // re-lowering on requiredDomain(), never retag values from the old graph.
  // A near match here fails; it must not turn into Tier 3 or Tier 4.
  NativeScaleAddPlan finish(const NativeScaleAddRequest &rebuilt) const;

private:
  friend NativeScaleAddResult lowerNativeScaleAdd(const NativeScaleAddRequest &,
                                                  const NativeStrictScaleSolver &);
  NativeStrictScaleRestart(const NativeScaleAddRequest &, ScaleDomain);
  struct Producer {
    EncodingCallReceipt source;
    std::optional<EncodingCallReceipt> plain;
    ScaleState result;
    bool operator==(const Producer &) const = default;
  };
  static Producer replay(const NativeScaleOperand &, const ScaleDomain &);
  ScaleDomain domain_;
  std::string result_, lhs_, rhs_;
  int level_, slots_;
  Producer left_{}, right_{};
};

} // namespace snabf::native
