#pragma once

#include "pass2/AffineErrorForm.h"
#include "pass3/PrimeInstantiator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace snabf {
using ExactRational = pass2::Exact;

struct ScaleDomain {
  std::string key_id;
  int ring_n = 0, default_log_scale = 0;
  std::vector<uint64_t> q, p; // level order, never sorted
  bool operator==(const ScaleDomain &) const = default;

  void check(int level) const {
    if (key_id.empty() || ring_n < 4 || !std::has_single_bit(unsigned(ring_n)) ||
        default_log_scale < 1 || default_log_scale > 128 || level < 0 ||
        size_t(level) >= q.size() || p.empty())
      throw std::invalid_argument("unbound scale domain/level");
    std::set<uint64_t> seen;
    for (const auto *basis : {&q, &p})
      for (auto v : *basis)
        if (v % (2ULL * ring_n) != 1 || !seen.insert(v).second || !pass3::prime(v))
          throw std::invalid_argument("invalid ordered Q/P");
  }
};

struct EncodingCallContext {
  std::string call_id, site_id;
  int operand_index = -1, level = -1, slots = 0;
  ScaleDomain domain;
  bool is_batched = false;
  std::string arg_type;
  int arg_precision_bits = 0, encoder_prec_bits = 0;
  // A secret input has no public bit string at compile time. It is not a
  // known empty/zero plaintext, nor is its planned call a runtime observation.
  bool dynamic_input = false;
  bool operator==(const EncodingCallContext &) const = default;
};

struct EncodingCallReceipt : EncodingCallContext {
  std::string branch, quantization_rule;
  size_t payload_count;
  std::string payload_sha256; // IEEE binary64 words, big endian, before periodic tiling
  std::optional<std::string> argument_sha256; // actual tiled public argument
  ExactRational requested_scale, realized_meta_scale, realized_eff_scale;
  bool operator==(const EncodingCallReceipt &) const = default;
};

// A homogeneous operand or an explicitly rebased reference. After a join
// there need not be a single effective gain for the mathematical sum.
struct ScaleState {
  ExactRational metadata, effective;
  bool operator==(const ScaleState &) const = default;
};
struct PrimeDivision {
  uint64_t q; ScaleState after;
  bool operator==(const PrimeDivision &) const = default;
};
struct RescaleStepReceipt {
  ScaleDomain domain;
  int in_level, out_level, mul_depth_delta, levels_consumed;
  std::vector<uint64_t> dropped_q;
  ExactRational meta_before, meta_after, eff_before, eff_after;
  std::vector<PrimeDivision> divisions;
  bool operator==(const RescaleStepReceipt &) const = default;
};

enum class JoinOp { Add, Sub };
struct AddSubScaleAnalysis {
  int64_t lhs_int_mul = 1, rhs_int_mul = 1;
  ExactRational out_meta_scale, lhs_signed_effect, rhs_signed_effect;
  ExactRational abs_lhs_residual, abs_rhs_residual;
  std::optional<ExactRational> common_effective_scale;
  std::optional<ExactRational> lhs_bound, rhs_bound;
  std::optional<ExactRational> encoded_bias, decoded_bias;
  std::optional<double> encoded_bias_upper_mpfr, decoded_bias_upper_mpfr;
  bool missing_operand_bounds;
};

struct ZeroDepthFoldResult {
  bool within_search_window, exact_representable;
  ExactRational rho, requested_pt_scale, realized_pt_meta, realized_pt_eff;
  ExactRational residual_scale_diff; // signed relative output effective gain: E'/target.E - 1
  ExactRational output_meta_scale, output_eff_scale;
  EncodingCallReceipt replacement;
};

class ExactScaleTracker {
public:
  // Build from the call site and public bits, not a claimed branch or M/E.
  // This small adapter also serves the released B2.1 encoding ABI without
  // adding a link dependency or a field to its sealed modules/audits.
  static EncodingCallReceipt encode(const EncodingCallContext &call,
                                    const ExactRational &requested,
                                    std::span<const double> payload) {
    call.domain.check(call.level);
    if (call.call_id.empty() || call.site_id.empty() || call.operand_index < 0 ||
        !call.is_batched || call.arg_type != "[]float64" || call.arg_precision_bits != 53 ||
        call.encoder_prec_bits < 53 || call.encoder_prec_bits > 128 || call.slots <= 0 ||
        !std::has_single_bit(unsigned(call.slots)) || call.slots > call.domain.ring_n / 2 ||
        call.dynamic_input != payload.empty() || (!payload.empty() && call.slots % payload.size()))
      throw std::invalid_argument("unbound encoding call/unsupported branch or payload");
    auto digest = [&](size_t count) {
      llvm::SHA256 h;
      for (size_t j = 0; j < count; ++j) {
        const double v = payload[j % payload.size()];
        if (!std::isfinite(v)) throw std::invalid_argument("nonfinite public payload");
        auto word = std::bit_cast<uint64_t>(v);
        uint8_t bytes[8];
        for (int i = 7; i >= 0; --i, word >>= 8) bytes[i] = uint8_t(word);
        h.update(llvm::ArrayRef<uint8_t>(bytes, 8));
      }
      return llvm::toHex(h.final(), true);
    };
    auto m = pass2::roundCKKSScale(requested);
    const bool f64 = call.encoder_prec_bits == 53;
    auto hash = digest(payload.size());
    std::optional<std::string> arg;
    if (!call.dynamic_input) arg = payload.size() == size_t(call.slots) ? hash : digest(call.slots);
    return {call, f64 ? "embed_double_f64" : "bigfloat_value",
            f64 ? "batched-f64" : "batched-bigfloat", payload.size(), hash, arg,
            requested, m, pass2::effectiveEncoderScale(m, call.encoder_prec_bits)};
  }

  static void verifyEncoding(const EncodingCallReceipt &claim, const EncodingCallContext &call,
                             const ExactRational &requested, std::span<const double> public_bits) {
    if (!(claim == encode(call, requested, public_bits)))
      throw std::invalid_argument("stale encoding call receipt");
  }

  static ScaleState multiply(const ScaleState &, const ScaleState &);
  // Only replay accepts PREC128. Production lowering must use the separate
  // entry point, which also rejects a one-Q request under a two-Q mode.
  static RescaleStepReceipt replayRescale(const ScaleState &, const ScaleDomain &, int in_level,
                                          std::span<const uint64_t> dropped_q);
  static void verifyReplayedRescale(const RescaleStepReceipt &claim, const ScaleState &in,
                                    const ScaleDomain &domain, int level, std::span<const uint64_t> dropped) {
    if (!(claim == replayRescale(in, domain, level, dropped)))
      throw std::invalid_argument("stale rescale step receipt");
  }
  static RescaleStepReceipt rescaleForLowering(const ScaleState &, const ScaleDomain &, int in_level,
                                               std::span<const uint64_t> dropped_q, int mul_depth_delta);
  static AddSubScaleAnalysis analyzeAddSubMismatch(JoinOp, const ScaleState &, const ScaleState &,
      const std::optional<ExactRational> &lhs_bound, const std::optional<ExactRational> &rhs_bound);
  // Arithmetic candidate only. Caller owns use-by-use compatibility and budget;
  // neither a search-window hit nor a rounded match authorizes an IR rewrite.
  static ZeroDepthFoldResult computeZeroDepthPlainFold(const ScaleState &ct, const ScaleState &target,
      const EncodingCallContext &, std::span<const double> unchanged_payload,
      const ExactRational &old_requested, std::span<const uint64_t> dropped_q);
};
} // namespace snabf
