#include "ExactScaleTracker.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

namespace snabf {
using R = ExactRational;

EncodingCallReceipt ExactScaleTracker::encode(const EncodingCallContext &call, const R &requested,
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

ZeroDepthFoldResult ExactScaleTracker::computeZeroDepthPlainFold(const ScaleState &ct,
    const ScaleState &target, const EncodingCallContext &call, std::span<const double> payload,
    const R &old_requested, std::span<const uint64_t> tail) {
  checkState(ct); checkState(target);
  if (call.dynamic_input || call.operand_index != 1)
    throw std::invalid_argument("fold requires an existing public plaintext operand");
  auto old = encode(call, old_requested, payload);
  // A candidate for a real rewrite must use the lowering gate, not replay-only multi-Q.
  (void)rescaleForLowering(ct, call.domain, call.level, tail, 0);
  const R requested = target.effective * R(std::to_string(tail.front())) / ct.effective;
  const R rho = requested / old.realized_eff_scale;
  auto pt = encode(call, requested, payload);
  auto product = multiply(ct, {pt.realized_meta_scale, pt.realized_eff_scale});
  auto out = rescaleForLowering(product, call.domain, call.level, tail, 0);
  return {rho >= R(1,2) && rho <= 2, requested == pt.realized_meta_scale &&
          requested == pt.realized_eff_scale && out.meta_after == target.metadata &&
          out.eff_after == target.effective, rho, requested, pt.realized_meta_scale,
          pt.realized_eff_scale, out.eff_after / target.effective - 1,
          out.meta_after, out.eff_after, std::move(pt)};
}
} // namespace snabf
