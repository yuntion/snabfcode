#include "NoiseCertVerifier.h"

#include <bit>
#include <mpfr.h>

namespace snabf::ir::noise {
namespace {
// Only used away from the binary64 underflow/overflow boundaries below.
Exact rn53(const Exact &x) {
  mpfr_t f;
  mpfr_init2(f, 53);
  mpfr_set_q(f, x.get_mpq_t(), MPFR_RNDN);
  Exact result;
  mpfr_get_q(result.get_mpq_t(), f);
  mpfr_clear(f);
  return result;
}

// DirectGo v1 tiles the public argument BEFORE calling Encode([]float64).
// For a constant real tile every non-DC IFFT butterfly multiplies zero by
// a finite root. The DC path doubles exactly, divides by slots, and stays at
// coefficient zero through bit reversal and the sparse Y -> X embedding.
// See v6.2.0 ckks/{encoder,ckks_vector_ops,utils}.go and rlwe/utils.go.
const char *constant(const Node &n, const Domain &d, LocalTransfer &row) {
  if (!n.bound_plain_call || !n.effective_encoder_scale) return "missing_bound_encoding_call";
  const auto &call = n.bound_plain_call->receipt();
  if (call.call_id != "encode:" + n.id || call.site_id != n.id || call.operand_index != 1 || call.dynamic_input ||
      !call.argument_sha256 || call.level != n.level || call.domain.key_id != d.key_id || call.domain.ring_n != d.ring_n ||
      call.domain.q != d.q || call.domain.p != d.p || call.realized_meta_scale != n.encoded_scale ||
      call.realized_eff_scale != n.effective_encoder_scale || call.payload_count != (n.real_scalar ? 1 : n.plain_values.size()) ||
      call.encoder_prec_bits != std::max(53, call.domain.default_log_scale))
    throw std::invalid_argument(n.id + ": B2-i mismatched encoding use");
  row.argument_sha256 = *call.argument_sha256;
  if (call.branch != "embed_double_f64" || call.slots < 16) return "unsupported_encoding_branch_or_slots";
  if (n.op == Op::VectorMulPlain || call.payload_count != 1) return "vector_ifft_not_proved";
  const double value = n.real_scalar ? *n.real_scalar : n.plain_values.front();
  if (n.bound_plain_call->singletonBits() != std::bit_cast<uint64_t>(value))
    throw std::invalid_argument(n.id + ": B2-i mismatched singleton bits");
  const Exact w = pass2::exactBinary(value);
  const Exact a = abs(w), product = a * *n.effective_encoder_scale;
  // Narrow normal-only proof. Limit the fixed-point path to the small uint64
  // branch and canonical NTT inputs; no host conversion of an oversized float.
  if ((a != 0 && (a < Exact(0x1p-512) || a > Exact(0x1p100) || product < Exact(0x1p-512))) ||
      product >= Exact(0x1p52)) return "unsupported_fixed_point_range";
  const Exact h(1,2);
  auto integer = [](const Exact &v) -> mpz_class { return v.get_num() / v.get_den(); };
  mpz_class c = integer(rn53(rn53(product) + h));
  if (c != integer(rn53(product + h))) return "ambiguous_fused_fixed_point";
  for (int l = 0; l <= n.level; ++l) {
    const mpz_class q(std::to_string(d.q[l]));
    if (d.q[l] >= (uint64_t(1) << 60)) return "ntt_modulus_outside_proved_range";
    // Negative fast CRT reduces c first. In the positive small-c branch the
    // implementation actually writes c, NOT c % qi. Don't silently reduce it.
    if (w > 0 && c >= q) return "noncanonical_positive_fast_crt_limb";
    if (w < 0 && c != q && c % q == 0) return "noncanonical_negative_fast_crt_limb";
  }
  if (2 * c >= row.active_q) return "plaintext_centered_lift_unproved";
  if (w < 0) c = -c;
  row.encoded_constant = Exact(c);
  row.delta_pt = *row.encoded_constant - *n.effective_encoder_scale * w;
  return nullptr;
}
} // namespace

LocalStep boundedStep(const Topology &g, size_t i, const std::vector<LocalStep> &done,
    const std::optional<Exact> &a, const std::optional<Exact> &ideal,
    bool scale_bound, const Exact &ma, const Exact &mb) {
  const auto &n = g.nodes.at(i);
  const auto &d = g.domains.at(n.domain);
  Exact q = 1;
  for (int l = 0; l <= n.level; ++l) q *= Exact(std::to_string(d.q[l]));
  LocalStep out{(q - 1) / 2, {}};
  auto row = [&](Role role) -> LocalTransfer & {
    auto &r = out.rows.emplace_back();
    r.site = "node/" + n.id + "/" + name(role);
    r.active_q = q;
    for (auto j : n.inputs) r.inputs.push_back(g.nodes.at(j).id);
    return r;
  };
  auto coeff = [&](size_t j) -> const Exact & { return done.at(n.inputs.at(j)).centered_coeff_abs; };
  auto lift = [&](LocalTransfer &r, const Exact &candidate) {
    r.candidate_coeff_abs = candidate;
    if (2 * candidate >= q) { r.blocker = "need_tighter_centered_decryption_coefficients"; return; }
    out.centered_coeff_abs = candidate;
    r.formula = "b2i/centered-linear-lift-v1";
    r.bias = 0; // strict interval proves the centered representative is unchanged
  };
  switch (n.op) {
  case Op::Add: case Op::Sub: {
    auto &r = row(n.op == Op::Add ? Role::Add : Role::Sub);
    if (!scale_bound) { r.blocker = "missing_alignment"; break; }
    r.lhs_gain = ma; r.rhs_gain = n.op == Op::Add ? mb : -mb;
    if (n.inputs[0] == n.inputs[1]) lift(r, abs(ma + *r.rhs_gain) * coeff(0));
    else lift(r, ma * coeff(0) + mb * coeff(1));
    break;
  }
  case Op::Neg: case Op::Drop: {
    auto &r = row(n.op == Op::Neg ? Role::Neg : Role::Drop);
    r.lhs_gain = n.op == Op::Neg ? -1 : 1;
    lift(r, coeff(0)); // q is the DESTINATION product, including q[0]
    break;
  }
  case Op::ScalarMulPlain: case Op::VectorMulPlain: case Op::AddPlain: {
    bool add = n.op == Op::AddPlain;
    auto &enc = row(Role::PlainEncode);
    auto why = constant(n, d, enc);
    if (why) enc.blocker = why;
    else if (!scale_bound || !a || (!add && !ideal)) enc.blocker = "missing_metadata_reference_or_ideal_bound";
    else {
      // C*(A*x+e)-S*w*x = C*e + A*(C-E*w)*x + (A*E-S)*w*x.
      // The LAST summand (or its two-sided join analogue) is already in
      // replayPhysical. C carries delta*e, signed, even for shared-key terms.
      enc.formula = "b2i/constant-real-tile-v6.2.0";
      enc.lhs_gain = add ? ma : *enc.encoded_constant;
      enc.bias = abs(*enc.delta_pt) * (add ? mb : Exact(*a * *ideal));
    }
    const auto c = enc.encoded_constant;
    auto &op = row(add ? Role::PlainAdd : Role::PlainMul);
    if (!c) op.blocker = "unproved_plaintext_polynomial";
    else if (add && !scale_bound) op.blocker = "missing_alignment";
    else {
      op.lhs_gain = add ? ma : *c;
      if (add) op.rhs_gain = mb;
      lift(op, add ? Exact(ma * coeff(0) + mb * abs(*c)) : Exact(abs(*c) * coeff(0)));
    }
    break;
  }
  default: break;
  }
  return out;
}
} // namespace snabf::ir::noise
