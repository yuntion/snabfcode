#include "NativeScaleLower.h"
#include <algorithm>
#include <array>
#include <bit>
#include <initializer_list>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace snabf::native {
namespace {
using R = ExactRational;
using T = ExactScaleTracker;

void need(bool ok, const char *reason) {
  if (!ok) throw std::invalid_argument(reason);
}

void checkMagnitude(const std::optional<R> &x) {
  if (x) need(x->get_den() > 0 && x->get_num() >= 0 &&
              gcd(x->get_num(), x->get_den()) == 1, "invalid decoded bound or budget");
}

void checkValue(const NativeScaleValue &v) {
  need(v.domain.default_log_scale <= 64, "UNSUPPORTED_MULTIPRIME_RESCALE");
  v.domain.check(v.level);
  need(!v.id.empty() && v.slots > 0 && std::has_single_bit(unsigned(v.slots)) &&
       v.slots <= v.domain.ring_n / 2, "unbound ciphertext or slots");
  // Reuse the tracker's state validation before doing any rational division.
  (void)T::multiply(v.scale, {1, 1});
}

void checkRequest(const NativeScaleAddRequest &r) {
  const auto &l = r.lhs.value, &h = r.rhs.value;
  checkValue(l);
  checkValue(h);
  need(l.domain == h.domain && l.level == h.level && l.slots == h.slots,
       "native add needs identical ordered Q/P, key, ring, level and slots");
  need(!r.result_id.empty() && r.result_id != l.id && r.result_id != h.id &&
       (l.id != h.id || l.scale == h.scale), "inconsistent add SSA bindings");
  need(r.min_output_level >= 0 && r.min_output_level <= l.level, "invalid output level floor");
  checkMagnitude(r.lhs.abs_bound);
  checkMagnitude(r.rhs.abs_bound);
  checkMagnitude(r.decoded_bias_budget);
}

bool exactJoin(const ScaleState &a, const ScaleState &b) {
  // Do not waive an existing E/M drift, even when it is common to both sides.
  return a == b && a.effective == a.metadata;
}

bool near(const R &a, const R &b) {
  // Symmetric, conservative relative pre-filter (no logs or host doubles).
  return abs(a - b) * 100000000 <= std::min(a, b);
}

AddSubScaleAnalysis analyze(const NativeScaleAddRequest &r, const ScaleState &a, const ScaleState &b) {
  return T::analyzeAddSubMismatch(JoinOp::Add, a, b, r.lhs.abs_bound, r.rhs.abs_bound);
}

std::optional<AddSubScaleAnalysis> proveJoin(const NativeScaleAddRequest &r,
                                            const ScaleState &a, const ScaleState &b) {
  const bool exact = exactJoin(a, b);
  if (!exact && (!near(a.metadata, b.metadata) || !near(a.effective, b.effective) ||
                 !r.lhs.abs_bound || !r.rhs.abs_bound)) return std::nullopt;
  auto proof = analyze(r, a, b);
  if (!exact && (!proof.decoded_bias || *proof.decoded_bias > r.decoded_bias_budget.value_or(0)))
    return std::nullopt;
  return proof;
}

NativeScaleAddPlan plan(const NativeScaleAddRequest &r, NativeScaleMode mode,
                        const ScaleState &a, const ScaleState &b, AddSubScaleAnalysis proof) {
  // Without bounds, only identically zero coefficients authorize a zero charge.
  need(proof.decoded_bias || (proof.abs_lhs_residual == 0 && proof.abs_rhs_residual == 0),
       "missing two-sided decoded bias proof");
  R charge = proof.decoded_bias.value_or(0);
  return {mode, r.lhs.value.domain, r.lhs.value.slots, {},
          {r.result_id, r.lhs.value.id, r.rhs.value.id, r.lhs.value.level, a, b, std::move(proof)},
          std::nullopt, std::move(charge), 0, 0, {}};
}

void checkCall(const EncodingCallContext &c, const NativeScaleValue &ct) {
  need(!c.dynamic_input && c.operand_index == 1 && c.domain == ct.domain &&
       c.level == ct.level && c.slots == ct.slots, "public multiply encoding binding mismatch");
}

bool distinct(std::initializer_list<std::string_view> names) {
  std::set<std::string_view> seen;
  for (auto name : names) if (name.empty() || !seen.insert(name).second) return false;
  return true;
}

bool soleUse(const std::vector<NativeScaleUse> &uses, const std::string &id, int operand) {
  return uses.size() == 1 && uses.front() == NativeScaleUse{id, operand};
}

struct ProductPath {
  NativeMulPlain mul;
  NativeRescaleOne rescale;
  ScaleState after() const { return {rescale.step.meta_after, rescale.step.eff_after}; }
};

ProductPath productPath(const NativeScaleValue &ct, const EncodingCallReceipt &pt,
                        const std::string &out, std::span<const uint64_t> tail, int depth) {
  auto product = T::multiply(ct.scale, {pt.realized_meta_scale, pt.realized_eff_scale});
  auto step = T::rescaleForLowering(product, ct.domain, ct.level, tail, depth);
  return {{pt.site_id, ct.id, pt.call_id, std::move(product)},
          {out, pt.site_id, std::move(step)}};
}

void emitPath(NativeScaleAddPlan &p, NativeEncodePlain pt, ProductPath path) {
  p.preparation.emplace_back(std::move(pt));
  p.preparation.emplace_back(std::move(path.mul));
  p.preparation.emplace_back(std::move(path.rescale));
}

void verifyPublicMul(const NativeScaleOperand &side) {
  const auto &f = *side.public_mul;
  checkValue(f.input);
  checkCall(f.call, f.input);
  need(f.input.domain == side.value.domain && f.input.slots == side.value.slots &&
       f.input.level > 0 && f.input.level - 1 == side.value.level,
       "fold is not an existing same-domain multiply/rescale at this operand");
  need(distinct({f.input.id, f.call.call_id, f.call.site_id, side.value.id}),
       "aliased public multiply SSA bindings");
  T::verifyEncoding(f.original_encoding, f.call, f.requested, f.logical_payload);
  auto old = productPath(f.input, f.original_encoding, side.value.id, f.dropped_q, 0);
  need(old.after() == side.value.scale, "stale existing multiply/rescale output");
}

std::optional<NativeScaleAddPlan> tryFold(const NativeScaleAddRequest &r, bool left) {
  const auto &side = left ? r.lhs : r.rhs, &peer = left ? r.rhs : r.lhs;
  if (!side.public_mul || side.value.id == peer.value.id) return std::nullopt;
  const auto &f = *side.public_mul;
  verifyPublicMul(side);
  need(distinct({f.input.id, f.call.call_id, f.call.site_id, side.value.id, peer.value.id,
                 r.result_id}), "aliased fold SSA bindings");
  if (!soleUse(f.plain_uses, f.call.site_id, 1) ||
      !soleUse(f.product_uses, side.value.id, 0) ||
      !soleUse(f.rescale_uses, r.result_id, left ? 0 : 1)) return std::nullopt;
  if (peer.public_mul) {
    const auto &other = *peer.public_mul;
    for (const auto *edited : {&f.call.call_id, &f.call.site_id, &side.value.id})
      if (*edited == other.input.id || *edited == other.call.call_id || *edited == other.call.site_id)
        return std::nullopt;
  }

  auto fold = T::computeZeroDepthPlainFold(f.input.scale, peer.value.scale, f.call,
                                          f.logical_payload, f.requested, f.dropped_q);
  if (!fold.within_search_window) return std::nullopt;
  auto path = productPath(f.input, fold.replacement, side.value.id, f.dropped_q, 0);
  const auto after = path.after();
  if (after == side.value.scale) return std::nullopt; // cannot repair this join
  const auto &a = left ? after : peer.value.scale, &b = left ? peer.value.scale : after;
  auto proof = proveJoin(r, a, b);
  if (!proof) return std::nullopt;
  auto p = plan(r, NativeScaleMode::FoldIntoPlain, a, b, std::move(*proof));
  emitPath(p, {fold.replacement, f.logical_payload}, std::move(path));
  p.fold = std::move(fold);
  return p;
}

std::optional<NativeScaleAddPlan> tryExplicit(const NativeScaleAddRequest &r, bool left) {
  const auto &side = left ? r.lhs : r.rhs, &peer = left ? r.rhs : r.lhs;
  const auto &ct = side.value, &target = peer.value;
  if (!side.align_one || ct.level <= r.min_output_level || ct.id == target.id) return std::nullopt;
  const auto &a = *side.align_one;
  checkCall(a.call, ct);
  need(distinct({a.call.call_id, a.call.site_id, a.rescale_id, a.peer_drop_id,
                 ct.id, target.id, r.result_id}), "explicit alignment needs fresh SSA bindings");
  for (const auto *operand : {&side, &peer}) {
    if (!operand->public_mul) continue;
    const auto &f = *operand->public_mul;
    for (const auto *fresh : {&a.call.call_id, &a.call.site_id, &a.rescale_id, &a.peer_drop_id})
      need(*fresh != f.input.id && *fresh != f.call.call_id && *fresh != f.call.site_id,
           "explicit alignment would overwrite an existing definition");
  }

  const std::array<uint64_t, 1> tail{ct.domain.q[ct.level]};
  const R q(std::to_string(tail.front()));
  const R requested = target.scale.effective * q / ct.scale.effective;
  const std::vector<double> one{1.0}; // tile across the actual encoded slots
  auto pt = T::encode(a.call, requested, one);
  auto path = productPath(ct, pt, a.rescale_id, tail, 1);
  const auto after = path.after();
  // This narrow fallback admits only a physically exact repair. Never write
  // target metadata onto the product, the rescaled value, or the peer.
  if (!exactJoin(after, target.scale)) return std::nullopt;
  const auto &l = left ? after : target.scale, &h = left ? target.scale : after;
  auto p = plan(r, NativeScaleMode::ExplicitAlignMul, l, h, analyze(r, l, h));
  const auto &step = path.rescale.step;
  p.mul_depth_delta = step.mul_depth_delta;
  p.levels_consumed = step.levels_consumed;
  p.dropped_q = step.dropped_q;
  p.add.level = step.out_level;
  p.add.lhs = left ? a.rescale_id : a.peer_drop_id;
  p.add.rhs = left ? a.peer_drop_id : a.rescale_id;
  emitPath(p, {std::move(pt), one}, std::move(path));
  p.preparation.emplace_back(NativeDropLevel{a.peer_drop_id, target.id, target.level,
                                             p.add.level, target.scale});
  return p;
}
} // namespace

NativeStrictScaleRestart::Producer NativeStrictScaleRestart::replay(const NativeScaleOperand &side,
                                                                     const ScaleDomain &basis) {
  need(side.source_encoding.has_value(), "strict re-solve needs source-bound producer replay");
  const auto &old = *side.source_encoding;
  const auto &root = side.public_mul ? side.public_mul->input : side.value;
  need(old.dynamic_input && old.operand_index == 0 && old.site_id == root.id &&
       old.call_id != root.id && old.domain == root.domain && old.level == root.level &&
       old.slots == root.slots, "strict re-solve source binding mismatch");
  T::verifyEncoding(old, old, old.requested_scale, {});
  need(root.scale == ScaleState{old.realized_meta_scale, old.realized_eff_scale},
       "strict re-solve source physical scale mismatch");

  EncodingCallContext sourceCall = old;
  sourceCall.domain = basis;
  auto source = T::encode(sourceCall, old.requested_scale, {});
  ScaleState sourceScale{source.realized_meta_scale, source.realized_eff_scale};
  if (!side.public_mul) return {std::move(source), std::nullopt, std::move(sourceScale)};

  verifyPublicMul(side);
  const auto &f = *side.public_mul;
  auto call = f.call;
  call.domain = basis;
  auto pt = T::encode(call, f.requested, f.logical_payload);
  const NativeScaleValue input{root.id, basis, root.level, root.slots, std::move(sourceScale)};
  const std::array<uint64_t, 1> tail{basis.q.at(root.level)};
  auto path = productPath(input, pt, side.value.id, tail, 0);
  return {std::move(source), std::move(pt), path.after()};
}

NativeStrictScaleRestart::NativeStrictScaleRestart(const NativeScaleAddRequest &r, ScaleDomain d)
    : domain_(std::move(d)), result_(r.result_id), lhs_(r.lhs.value.id), rhs_(r.rhs.value.id),
      level_(r.lhs.value.level), slots_(r.lhs.value.slots) {
  const auto &old = r.lhs.value.domain;
  need(domain_.default_log_scale <= 64, "UNSUPPORTED_MULTIPRIME_RESCALE");
  domain_.check(level_);
  // This implementation supports a concrete Q re-solve, not a renamed key,
  // a P-only change or a claim that two live ciphertexts are at the same level.
  need(domain_.ring_n == old.ring_n && domain_.q != old.q, "strict re-solve needs a new ordered Q basis");
  left_ = replay(r.lhs, domain_);
  right_ = replay(r.rhs, domain_);
  need(exactJoin(left_.result, right_.result), "strict candidate does not solve the concrete scale equation");
}

NativeScaleAddPlan NativeStrictScaleRestart::finish(const NativeScaleAddRequest &r) const {
  checkRequest(r);
  const auto &l = r.lhs.value, &h = r.rhs.value;
  need(r.result_id == result_ && l.id == lhs_ && h.id == rhs_ && l.domain == domain_ &&
       l.level == level_ && l.slots == slots_, "strict re-solve does not bind this rebuilt join");
  need(exactJoin(l.scale, h.scale), "strict re-solve did not produce exact metadata and effective scales");
  need(replay(r.lhs, domain_) == left_ && replay(r.rhs, domain_) == right_,
       "strict re-solve changed an encoding, logical payload or producer");
  return plan(r, NativeScaleMode::ExactPrimeMatch, l.scale, h.scale, analyze(r, l.scale, h.scale));
}

NativeScaleAddResult lowerNativeScaleAdd(const NativeScaleAddRequest &r,
                                         const NativeStrictScaleSolver &strict_solver) {
  checkRequest(r);
  const auto &l = r.lhs.value.scale, &h = r.rhs.value.scale;
  if (exactJoin(l, h)) return plan(r, NativeScaleMode::None, l, h, analyze(r, l, h));
  for (bool left : {true, false})
    if (auto p = tryFold(r, left)) return std::move(*p);
  if (strict_solver && l != h && r.lhs.source_encoding && r.rhs.source_encoding)
    if (auto domain = strict_solver(r)) return NativeStrictScaleRestart(r, std::move(*domain));
  if (auto proof = proveJoin(r, l, h))
    return plan(r, NativeScaleMode::BoundedMismatchBias, l, h, std::move(*proof));
  for (bool left : {true, false})
    if (auto p = tryExplicit(r, left)) return std::move(*p);
  throw std::invalid_argument("unresolved native add: no authorized fold, strict match, bounded bias or exact one-Q alignment");
}

} // namespace snabf::native
