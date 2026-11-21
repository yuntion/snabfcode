#include "ExactScaleTracker.h"

namespace snabf {
namespace {
using R = ExactRational;
void need(bool ok, const char *reason) {
  if (!ok) throw std::invalid_argument(reason);
}
void magnitude(const R &q) {
  need(q.get_den() > 0 && gcd(q.get_num(), q.get_den()) == 1 && q.get_num() >= 0,
       "invalid exact magnitude");
}
} // namespace

void ExactScaleTracker::checkState(const ScaleState &v) {
  magnitude(v.effective);
  need(v.effective > 0 && pass2::roundCKKSScale(v.metadata) == v.metadata,
       "invalid effective gain or unrepresentable metadata");
}

ScaleState ExactScaleTracker::multiply(const ScaleState &a, const ScaleState &b) {
  checkState(a); checkState(b);
  return {pass2::roundCKKSScale(a.metadata * b.metadata), a.effective * b.effective};
}

ScaleState ExactScaleTracker::replayPrimeDivision(const ScaleState &in,
    std::span<const uint64_t> ordered_q, int level) {
  checkState(in);
  need(level > 0 && size_t(level) < ordered_q.size() && pass3::prime(ordered_q[level]),
       "invalid prime or level for rescale");
  const R q(std::to_string(ordered_q[level]));
  return {pass2::roundCKKSScale(in.metadata / q), in.effective / q};
}

RescaleStepReceipt ExactScaleTracker::replayRescale(const ScaleState &in, const ScaleDomain &d,
    int level, std::span<const uint64_t> tail) {
  d.check(level);
  const int count = d.default_log_scale <= 64 ? 1 : 2;
  need(tail.size() == size_t(count) && level >= count, "rescale mode/level mismatch");
  RescaleStepReceipt r{d, level, level-count, 0, count, {tail.begin(), tail.end()},
                       in.metadata, in.metadata, in.effective, in.effective, {}};
  for (auto prime : tail) {
    need(prime == d.q[level], "dropped_q is not the active ordered Q suffix");
    auto next = replayPrimeDivision({r.meta_after, r.eff_after}, d.q, level--);
    r.meta_after = next.metadata;
    r.eff_after = next.effective;
    r.divisions.push_back({prime, std::move(next)});
  }
  return r;
}

RescaleStepReceipt ExactScaleTracker::rescaleForLowering(const ScaleState &in, const ScaleDomain &d,
    int level, std::span<const uint64_t> tail, int depth) {
  need(d.default_log_scale <= 64 && tail.size() <= 1, "UNSUPPORTED_MULTIPRIME_RESCALE");
  need(depth == 0 || depth == 1, "invalid mul_depth_delta");
  auto r = replayRescale(in, d, level, tail);
  r.mul_depth_delta = depth;
  return r;
}

AddSubScaleAnalysis ExactScaleTracker::analyzeAddSubMismatch(JoinOp op, const ScaleState &a,
    const ScaleState &b, const std::optional<R> &ba, const std::optional<R> &bb) {
  checkState(a); checkState(b);
  need(op == JoinOp::Add || op == JoinOp::Sub, "unknown join operation");
  if (ba) magnitude(*ba);
  if (bb) magnitude(*bb);
  AddSubScaleAnalysis r{};
  r.out_meta_scale = std::max(a.metadata, b.metadata);
  // rlwe.Scale.Div (RN128), then big.Float.Int, NOT Scale.BigInt.
  R ratio = pass2::roundCKKSScale(r.out_meta_scale / std::min(a.metadata, b.metadata));
  mpz_class g = ratio.get_num() / ratio.get_den();
  if (g < 1 || g > (mpz_class(1) << 50)) throw UnsupportedAddAlignment();
  uint64_t multiplier = 0;
  mpz_export(&multiplier, nullptr, 1, sizeof(multiplier), 0, 0, g.get_mpz_t());
  if (a.metadata < b.metadata) r.lhs_int_mul = int64_t(multiplier);
  if (a.metadata > b.metadata) r.rhs_int_mul = int64_t(multiplier);
  R ea = (a.metadata < b.metadata ? g : mpz_class(1)) * a.effective;
  R eb = (b.metadata < a.metadata ? g : mpz_class(1)) * b.effective;
  if (ea == eb) r.common_effective_scale = ea;
  r.lhs_signed_effect = ea - r.out_meta_scale;
  r.rhs_signed_effect = eb - r.out_meta_scale;
  if (op == JoinOp::Sub) r.rhs_signed_effect = -r.rhs_signed_effect;
  r.abs_lhs_residual = abs(r.lhs_signed_effect);
  r.abs_rhs_residual = abs(r.rhs_signed_effect);
  r.lhs_bound = ba; r.rhs_bound = bb;
  r.missing_operand_bounds = !ba || !bb; // even a zero residual does not supply a bound
  if (!r.missing_operand_bounds) {
    r.encoded_bias = r.abs_lhs_residual * *ba + r.abs_rhs_residual * *bb;
    r.decoded_bias = *r.encoded_bias / r.out_meta_scale;
    r.encoded_bias_upper_mpfr = pass2::upperDouble(*r.encoded_bias);
    r.decoded_bias_upper_mpfr = pass2::upperDouble(*r.decoded_bias);
  }
  return r;
}

} // namespace snabf
