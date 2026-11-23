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

// For a period P, descending DIF first doubles equal halves exactly. Only the
// first P lanes survive (zero times a finite root stays zero). Bit reversal,
// split real/imag, then Y -> X^(N/(2S)) give stride N/(2P), at most 2P words.
// RequireDiagonalEncodingV1 checks every relevant twiddle component to 2^-48
// against integer half-angle enclosures, not a guessed math.Cos ulp bound.
const char *diagonal(const Node &n, const Domain &d, int slots, std::span<const double> v, LocalTransfer &r) {
  if (!n.diagonal_mask || n.op != Op::VectorMulPlain || slots > 32768 || d.ring_n > 65536 ||
      slots > d.ring_n/2 || v.size() > unsigned(slots) || slots % v.size()) return "vector_ifft_not_proved";
  unsigned p = v.size();
  auto equal = [](double x, double y) { return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y); };
  while (p > 1 && std::equal(v.begin(), v.begin()+p/2, v.begin()+p/2, equal)) p /= 2;
  if (p < 2 || p > 64) return "vector_ifft_not_proved";
  Exact w = 0;
  for (double x : v.first(p)) w = std::max(w, Exact(abs(pass2::exactBinary(x))));
  const Exact e = *n.effective_encoder_scale;
  if (w < Exact(0x1p-300) || w > Exact(0x1p60) || e < Exact(0x1p-300) || e > Exact(0x1p300))
    return "unsupported_diagonal_range";
  // In complex modulus the ideal inputs at live stage j have size <= 2^j W
  // (before normalization, with the same initial S/P factor). Divide error
  // by this envelope. A butterfly gives 1+r' <= (1+r)(1+2^-45): twiddle
  // error <= 2*2^-48, and sum/diff, 4 products, 2 adds (or FMA) cost <64u,
  // u=2^-53. Absolute underflow errors (<2^-1022 per op) and division by S
  // fit the slack since W>=2^-300. k<=6; (1+2^-45)^k-1 < k*2^-44.
  const Exact err = w * std::countr_zero(p) * Exact(0x1p-44);
  const Exact scaled = e*(w+err);
  if (scaled >= Exact(0x1p48)) return "unsupported_diagonal_fixed_point_range";
  // uint64(RN(RN(|c|*E)+.5)): the two RN errors sum to <= 1/16 here,
  // including the fused case. Rounding the final integer costs <= 9/16.
  const Exact quant(9,16), envelope = scaled + quant;
  const mpz_class c = envelope.get_num() / envelope.get_den();
  if (2*c >= r.active_q) return "plaintext_centered_lift_unproved";
  Exact peak = 0;
  for (int l=0; l<=n.level; ++l) {
    if (d.q[l] >= (uint64_t(1)<<60)) return "ntt_modulus_outside_proved_range";
    const mpz_class q(std::to_string(d.q[l]));
    // Positive small words can be unreduced; negative multiples can write q.
    const Exact u(std::max(q,c) + 2*q*std::bit_width(unsigned(slots)));
    if (u >= Exact(0x1p64)) return "unproved_ntt_word_range";
    peak = std::max(peak,u);
  }
  r.ntt_word_upper = peak;
  r.diagonal = DiagonalEncoding{int(p), d.ring_n/int(2*p), w, err, Exact(c), 2*p*(e*err+quant)};
  return nullptr;
}

// DirectGo v1 tiles the public argument BEFORE calling Encode([]float64).
// For a constant real tile every non-DC IFFT butterfly multiplies zero by
// a finite root. The DC path doubles exactly, divides by slots, and stays at
// coefficient zero through bit reversal and the sparse Y -> X embedding.
// See v6.2.0 ckks/{encoder,ckks_vector_ops,utils}.go and rlwe/utils.go.
const char *constant(const Node &n, const Domain &d, int slots, LocalTransfer &row) {
  if (!n.bound_plain_call || !n.effective_encoder_scale) return "missing_bound_encoding_call";
  const auto &call = n.bound_plain_call->receipt();
  const auto values = n.real_scalar ? std::span<const double>(&*n.real_scalar, 1) : std::span<const double>(n.plain_values);
  if (call.call_id != "encode:" + n.id || call.site_id != n.id || call.operand_index != 1 || call.dynamic_input ||
      !call.argument_sha256 || call.slots != slots || call.level != n.level || call.domain.key_id != d.key_id || call.domain.ring_n != d.ring_n ||
      call.domain.q != d.q || call.domain.p != d.p || call.realized_meta_scale != n.encoded_scale ||
      call.realized_eff_scale != n.effective_encoder_scale || values.empty() || !n.bound_plain_call->matches(values) ||
      call.encoder_prec_bits != std::max(53, call.domain.default_log_scale))
    throw std::invalid_argument(n.id + ": B2-i mismatched encoding use");
  row.argument_sha256 = *call.argument_sha256;
  if (call.branch != "embed_double_f64" || call.slots < 16) return "unsupported_encoding_branch_or_slots";
  const double value = values.front();
  if (!std::all_of(values.begin(), values.end(), [&](double v) {
        return std::bit_cast<uint64_t>(v) == std::bit_cast<uint64_t>(value);
      })) return diagonal(n, d, slots, values, row);
  const Exact w = pass2::exactBinary(value);
  const Exact a = abs(w), product = a * *n.effective_encoder_scale;
  // Narrow normal-only proof, restricted to the small uint64 fixed-point path.
  if ((a != 0 && (a < Exact(0x1p-512) || a > Exact(0x1p100) || product < Exact(0x1p-512))) ||
      product >= Exact(0x1p52)) return "unsupported_fixed_point_range";
  const Exact h(1,2);
  auto integer = [](const Exact &v) -> mpz_class { return v.get_num() / v.get_den(); };
  mpz_class c = integer(rn53(rn53(product) + h));
  if (c != integer(rn53(product + h))) return "ambiguous_fused_fixed_point";
  std::optional<Exact> nttUpper;
  for (int l = 0; l <= n.level; ++l) {
    const mpz_class q(std::to_string(d.q[l]));
    if (d.q[l] >= (uint64_t(1) << 60)) return "ntt_modulus_outside_proved_range";
    // SingleFloat64ToFixedPointCRT: the positive small word is NOT reduced;
    // negative nonzero input can write q (also when c==0). Zero input writes 0.
    mpz_class word = c;
    if (w < 0) word = q - (c > q ? mpz_class(c % q) : c);
    if (word >= q) {
      // Standard non-Montgomery NewPlaintext; full and sparse use log2(2*slots)
      // forward stages. MRedLazy(v,F) lies in [0,2q) for ANY word v and F<q.
      // A butterfly never exceeds U+2q (optional U-=4q only lowers it).
      // Prove ALL additions fit before using NTTStandard's final BRedAdd.
      const Exact upper(word + 2 * q * std::bit_width(unsigned(call.slots)));
      if (upper >= Exact(0x1p64)) return "unproved_ntt_word_range";
      if (!nttUpper || upper > *nttUpper) nttUpper = upper;
    }
  }
  if (2 * c >= row.active_q) return "plaintext_centered_lift_unproved";
  if (w < 0) c = -c;
  row.ntt_word_upper = nttUpper;
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
  LocalStep out{};
  out.centered_coeff_abs = (q - 1) / 2;
  auto row = [&](Role role) -> LocalTransfer & {
    auto &r = out.rows.emplace_back();
    r.site = "node/" + n.id + "/" + name(role);
    r.active_q = q;
    for (auto j : n.inputs) r.inputs.push_back(g.nodes.at(j).id);
    return r;
  };
  auto coeff = [&](size_t j) -> const Exact & { return done.at(n.inputs.at(j)).centered_coeff_abs; };
  auto dc = [&](size_t j) -> const std::optional<Exact> & { return done.at(n.inputs.at(j)).centered_dc; };
  auto lift = [&](LocalTransfer &r, const Exact &candidate, std::optional<Exact> before = {}) {
    r.candidate_coeff_abs = candidate;
    if (!before && candidate == 0) before = 0; // every coefficient vanishes
    if (before) {
      if (before->get_den() != 1 || abs(*before) > candidate) throw std::logic_error("inconsistent centered DC operand");
      // Whole polynomial known, not a bound or a guessed sign. Odd Q gives
      // a unique centered integer; its signed correction is constant in all slots.
      const mpz_class h = (q.get_num() - 1) / 2;
      mpz_class residue = before->get_num() + h;
      mpz_mod(residue.get_mpz_t(), residue.get_mpz_t(), q.get_num_mpz_t());
      out.centered_dc = Exact(residue - h);
      out.centered_coeff_abs = abs(*out.centered_dc);
      r.dc_before_center = *before; r.dc_after_center = *out.centered_dc;
      r.dc_center_correction = *out.centered_dc - *before;
      r.bias = abs(*r.dc_center_correction);
      r.formula = *r.bias == 0 ? "b2i/centered-linear-lift-v1" : "b2i/exact-dc-recenter-v1";
      return;
    }
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
    if (dc(0) && dc(1)) {
      const Exact k = ma * *dc(0) + *r.rhs_gain * *dc(1);
      lift(r, abs(k), k);
    } else if (n.inputs[0] == n.inputs[1]) lift(r, abs(ma + *r.rhs_gain) * coeff(0));
    else lift(r, ma * coeff(0) + mb * coeff(1));
    break;
  }
  case Op::Neg: case Op::Drop: {
    auto &r = row(n.op == Op::Neg ? Role::Neg : Role::Drop);
    r.lhs_gain = n.op == Op::Neg ? -1 : 1;
    std::optional<Exact> before;
    if (dc(0)) before = *r.lhs_gain * *dc(0);
    lift(r, coeff(0), before); // q is the DESTINATION product, including q[0]
    break;
  }
  case Op::ScalarMulPlain: case Op::VectorMulPlain: case Op::AddPlain: {
    bool add = n.op == Op::AddPlain;
    auto &enc = row(Role::PlainEncode);
    auto why = constant(n, d, g.direct_go_slots, enc);
    if (why) enc.blocker = why;
    else if (!scale_bound || !a || (!add && !ideal)) enc.blocker = "missing_metadata_reference_or_ideal_bound";
    else if (enc.diagonal) {
      enc.formula = "b2i/guarded-periodic-real-diagonal-v1";
      enc.bias = *a * *ideal * enc.diagonal->delta_inf;
    } else {
      // C*(A*x+e)-S*w*x = C*e + A*(C-E*w)*x + (A*E-S)*w*x.
      // The LAST summand (or its two-sided join analogue) is already in
      // replayPhysical. C carries delta*e, signed, even for shared-key terms.
      enc.formula = enc.ntt_word_upper ? "b2i/dc-word-bounded-ntt-v6.2.0" : "b2i/constant-real-tile-v6.2.0";
      enc.lhs_gain = add ? ma : *enc.encoded_constant;
      enc.bias = abs(*enc.delta_pt) * (add ? mb : Exact(*a * *ideal));
      if (!add) out.plain_mul_gain = enc.encoded_constant;
    }
    const auto c = enc.encoded_constant;
    const auto diag = enc.diagonal; // row() can reallocate the transfers vector
    auto &op = row(add ? Role::PlainAdd : Role::PlainMul);
    if (diag) {
      const Exact candidate = coeff(0) * (2*diag->period) * diag->coeff_abs;
      op.candidate_coeff_abs = candidate;
      if (2*candidate < q) out.centered_coeff_abs = candidate;
      // Even when convolution fits, C*e is a transformed parent, not a
      // signed scalar. Neither this role nor any common event is discharged.
      op.blocker = 2*candidate < q ? "unproved_vector_parent_error" : "need_tighter_centered_decryption_coefficients";
    } else if (!c) op.blocker = "unproved_plaintext_polynomial";
    else if (add && !scale_bound) op.blocker = "missing_alignment";
    else {
      op.lhs_gain = add ? ma : *c;
      if (add) op.rhs_gain = mb;
      if (dc(0)) {
        const Exact k = add ? Exact(ma * *dc(0) + mb * *c) : Exact(*c * *dc(0));
        lift(op, abs(k), k);
      } else lift(op, add ? Exact(ma * coeff(0) + mb * abs(*c)) : Exact(abs(*c) * coeff(0)));
    }
    break;
  }
  default: break;
  }
  return out;
}
} // namespace snabf::ir::noise
