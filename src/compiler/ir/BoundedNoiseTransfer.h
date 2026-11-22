#pragma once

#include "ExactScaleTracker.h"
#include <optional>
#include <vector>

namespace snabf::ir::noise {
// Only constructed from the v1 emitter's freshly recomputed call. No proof,
// failure probability, centered envelope or rounded coefficient is an IR input.
class BoundPlainCall {
  EncodingCallReceipt call_;
  std::optional<uint64_t> singleton_;
  BoundPlainCall(EncodingCallReceipt call, std::optional<uint64_t> bits)
      : call_(std::move(call)), singleton_(bits) {}
public:
  static BoundPlainCall encode(const EncodingCallContext &ctx, const ExactRational &requested,
                               std::span<const double> values) {
    return {ExactScaleTracker::encode(ctx, requested, values),
            values.size() == 1 ? std::optional{std::bit_cast<uint64_t>(values[0])} : std::nullopt};
  }
  const EncodingCallReceipt &receipt() const { return call_; }
  std::optional<uint64_t> singletonBits() const { return singleton_; }
};

struct LocalTransfer {
  std::string site, formula, blocker, argument_sha256;
  std::vector<std::string> inputs;
  pass2::Exact active_q;
  std::optional<pass2::Exact> candidate_coeff_abs, lhs_gain, rhs_gain,
      encoded_constant, delta_pt, bias;
};
struct Topology;
struct LocalStep {
  // Always bounds the actual centered DECRYPTED polynomial coefficients, not
  // an ideal slot magnitude. Even an arbitrary residue has the full half-Q bound.
  pass2::Exact centered_coeff_abs;
  std::vector<LocalTransfer> rows;
};
LocalStep boundedStep(const Topology &, size_t index, const std::vector<LocalStep> &parents,
    const std::optional<pass2::Exact> &parent_scale, const std::optional<pass2::Exact> &parent_ideal,
    bool scale_bound, const pass2::Exact &ma, const pass2::Exact &mb);
} // namespace snabf::ir::noise
