#pragma once

#include "ExactScaleTracker.h"
#include <optional>
#include <vector>

namespace snabf::ir::noise {
// Only constructed from the v1 emitter's freshly recomputed call. No proof,
// failure probability, centered envelope or rounded coefficient is an IR input.
class BoundPlainCall {
  EncodingCallReceipt call_;
  std::vector<uint64_t> words_; // pre-tiling, including the sign bit of zero
  BoundPlainCall(EncodingCallReceipt call, std::span<const double> values)
      : call_(std::move(call)) {
    words_.reserve(values.size());
    for (double v : values) words_.push_back(std::bit_cast<uint64_t>(v));
  }
public:
  static BoundPlainCall encode(const EncodingCallContext &ctx, const ExactRational &requested,
                               std::span<const double> values) {
    return {ExactScaleTracker::encode(ctx, requested, values), values};
  }
  const EncodingCallReceipt &receipt() const { return call_; }
  bool matches(std::span<const double> values) const {
    return std::equal(words_.begin(), words_.end(), values.begin(), values.end(),
                      [](uint64_t w, double v) { return w == std::bit_cast<uint64_t>(v); });
  }
};

struct LocalTransfer {
  std::string site, formula, blocker, argument_sha256;
  std::vector<std::string> inputs;
  pass2::Exact active_q;
  std::optional<pass2::Exact> candidate_coeff_abs, lhs_gain, rhs_gain,
      encoded_constant, delta_pt, bias;
  std::optional<pass2::Exact> dc_before_center, dc_after_center, dc_center_correction, ntt_word_upper;
};
struct Topology;
struct LocalStep {
  // Always bounds the actual centered DECRYPTED polynomial coefficients, not
  // an ideal slot magnitude. Even an arbitrary residue has the full half-Q bound.
  pass2::Exact centered_coeff_abs;
  // Engaged only if the ENTIRE centered decryption polynomial is this constant.
  std::optional<pass2::Exact> centered_dc;
  std::optional<pass2::Exact> plain_mul_gain; // signed, same component, proved at this node
  std::vector<LocalTransfer> rows;
};
LocalStep boundedStep(const Topology &, size_t index, const std::vector<LocalStep> &parents,
    const std::optional<pass2::Exact> &parent_scale, const std::optional<pass2::Exact> &parent_ideal,
    bool scale_bound, const pass2::Exact &ma, const pass2::Exact &mb);
} // namespace snabf::ir::noise
