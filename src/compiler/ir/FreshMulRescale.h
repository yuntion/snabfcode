#pragma once

#include "NoiseCertVerifier.h"

namespace snabf::ir::noise {
// This is a support calculation on the *same* keys, not a noise sampler or
// a transfer of a conditional MGF. See B2FreshMulRescale.md for identities.
inline void freshMulRescale(const Topology &g, size_t i, const std::vector<LocalStep> &done,
    const std::optional<Exact> &a, const std::optional<Exact> &ideal, bool scaleBound,
    const Exact &ma, const Exact &mb, LocalStep &out) {
  const auto &n = g.nodes.at(i);
  const auto &d = g.domains.at(n.domain);
  auto word = [](uint64_t w) { return Exact(std::to_string(w)); };
  auto floor = [](const Exact &x) -> Exact { return Exact(mpz_class(x.get_num()/x.get_den())); }; // nonnegative
  const int N = d.ring_n, width = g.fresh_width;
  if (g.domains.size() != 1 || d.p.size() != 1 || d.q.size() > 8 || d.rescale_primes != 1 ||
      width < 1 || width > 64 || !std::has_single_bit(unsigned(width)) || N < 1024 || N > 65536 ||
      g.direct_go_slots < 16 || g.direct_go_slots > N/2 || !std::has_single_bit(unsigned(g.direct_go_slots)) || g.direct_go_slots % width ||
      d.p[0] >= (uint64_t(1)<<50) || std::any_of(d.q.begin(),d.q.end(), [](auto p){ return p >= (uint64_t(1)<<50); })) return;
  const Exact P = word(d.p[0]);
  Exact Q = 1;
  for (int j=0; j<=n.level; ++j) Q *= word(d.q[j]);
  auto row = [&](Role role) -> LocalTransfer & {
    const auto id = "node/" + n.id + "/" + name(role);
    for (auto &r : out.rows) if (r.site == id) return r;
    auto &r = out.rows.emplace_back(); r.site = id; r.active_q = Q;
    for (auto j : n.inputs) r.inputs.push_back(g.nodes[j].id);
    return r;
  };
  auto profile = [&](LocalTransfer &r) -> auto & {
    auto &m = r.premises;
    m["target"] = "epochs.RequireFreshMulRescaleV1/EvalFresh";
    m["key_id"] = d.key_id;
    m["P"] = P.get_str();
    m["source_id"] = g.nodes.front().id;
    return m;
  };
  auto set = [](auto &m, const char *k, const Exact &v) { m[k] = v.get_str(); };
  if (n.op == Op::Source) {
    auto &r = row(Role::Encrypt); auto &m = profile(r);
    r.blocker = "fresh_pk_support_is_not_a_conditional_law";
    if (i || n.input_draw != n.id || !n.bound_plain_call || n.source_bound != 1 || !n.encoded_scale || !n.effective_encoder_scale ||
        n.level+1 != int(d.q.size())) return;
    const auto &c = n.bound_plain_call->receipt();
    const Exact E = *n.effective_encoder_scale;
    if (!c.dynamic_input || c.site_id != n.id || c.call_id != "encode:"+n.id || c.operand_index != 0 || c.level != n.level ||
        c.slots != g.direct_go_slots || c.domain.q != d.q || c.domain.p != d.p || c.domain.key_id != d.key_id || c.domain.ring_n != N ||
        !c.is_batched || c.arg_type != "[]float64" || c.arg_precision_bits != 53 || c.encoder_prec_bits != 53 ||
        c.branch != "embed_double_f64" || c.payload_count || c.argument_sha256 || c.realized_meta_scale != *n.encoded_scale ||
        c.realized_eff_scale != E || c.domain.default_log_scale < 1 || c.domain.default_log_scale >= 47 ||
        c.requested_scale != Exact(mpz_class(1)<<c.domain.default_log_scale) || c.requested_scale != E || E != *n.encoded_scale) return;
    const Exact err = Exact(std::countr_zero(unsigned(width))) * Exact(0x1p-44);
    const Exact unit = E*err + Exact(9,16), U = floor(E+unit);
    const Exact pk = floor((19*(2*N+1) + (N+1)*(P-1)/2)/P);
    const Exact B = U+pk;
    set(m,"input_width",width); set(m,"encoder_scale",E); set(m,"plain_coeff_abs",U); set(m,"pk_noise_coeff_abs",pk);
    if (2*B >= Q) return;
    out.fresh_coeff = B;
    out.fresh_slot_error = Exact(2*width*unit + N*pk + abs(E-*n.encoded_scale));
    set(m,"output_coeff_abs",B); set(m,"slot_error_abs",*out.fresh_slot_error);
    return;
  }
  if (n.inputs.empty()) return;
  const auto &parent = done.at(n.inputs[0]);
  const auto &B = parent.fresh_coeff;
  if (n.op == Op::Rotate) {
    auto &r = row(Role::KeySwitch); auto &m = profile(r);
    r.blocker = "rotation_support_is_not_an_evk_error_contract";
    m["evk"] = n.evk;
    if (!B || n.k > 1 || n.evk != d.key_id+":rotation:"+std::to_string(n.rotation)) return;
    Exact digits=0;
    for (int j=n.k; j<=n.level; ++j) digits += word(d.q[j]);
    const Exact prefix = n.k ? word(d.q[0]) : Exact(1);
    const Exact inc = N*(prefix-1)/2 + floor((19*N*digits + (N+1)*(P-1)/2)/P);
    set(m,"parent_coeff_abs",*B); set(m,"retained_digit_sum_abs",digits); set(m,"prefix_product",prefix);
    set(m,"switch_delta_coeff_abs",inc);
    if (2*(*B+inc) >= Q) return;
    out.fresh_coeff = *B+inc; set(m,"output_coeff_abs",*out.fresh_coeff);
    if (parent.fresh_slot_error) {
      out.fresh_slot_error = *parent.fresh_slot_error + N*inc; set(m,"slot_error_abs",*out.fresh_slot_error);
    }
    return;
  }
  if (n.op == Op::Add || n.op == Op::Sub) {
    const auto &b2=done.at(n.inputs.at(1)).fresh_coeff;
    if (scaleBound && B && b2) {
      const Exact candidate = ma * *B + mb * *b2;
      if (2*candidate < Q) out.fresh_coeff = candidate;
    }
    return;
  }
  if (n.op == Op::Rescale) {
    auto &r = row(Role::RescaleRound); auto &m = profile(r);
    const int in=g.nodes.at(n.inputs[0]).level;
    if (in != n.level+1 || !scaleBound) { r.blocker="unbound_single_prime_rescale"; return; }
    const Exact p=word(d.q.at(in)), correction=(N+1)*(p-1)/2;
    set(m,"dropped_q",p); set(m,"input_q_product",Q*p); set(m,"same_key_remainder_abs",correction);
    set(m,"max_parent_coeff",p*(Q+1)/2-1-correction);
    if (!B) { r.blocker="missing_same_key_degree_one_parent_support"; return; }
    set(m,"parent_coeff_abs",*B);
    const Exact c=floor((*B+correction)/p); r.candidate_coeff_abs=c;
    if (2*c >= Q) { r.blocker="rescaled_centered_lift_unproved"; return; }
    out.fresh_coeff=c; r.lhs_gain=1/p; r.bias=N*correction/p;
    r.formula="b2j/same-key-one-prime-rescale-v1";
    return;
  }
  if (n.op != Op::VectorMulPlain && n.op != Op::ScalarMulPlain) return;
  const auto &enc=row(Role::PlainEncode);
  if (enc.formula.empty()) return;
  if (!enc.diagonal) {
    if (B && enc.encoded_constant) {
      const Exact c=abs(*enc.encoded_constant)* *B;
      if (2*c<Q) out.fresh_coeff=c;
    }
    return;
  }
  const auto diag=*enc.diagonal;
  const Exact L=2*diag.period*diag.coeff_abs;
  if (L == 0) return; // not this support family
  auto &r=row(Role::PlainMul); auto &m=profile(r);
  const Exact C=std::min(L,Exact(*n.effective_encoder_scale*diag.weight_abs+diag.delta_inf));
  set(m,"plain_l1",L); set(m,"plain_slot_abs",C); set(m,"max_parent_coeff",floor((Q-1)/(2*L)));
  r.candidate_coeff_abs.reset();
  if (!B || !a || !ideal) { r.blocker="missing_same_key_degree_one_parent_support"; return; }
  set(m,"parent_coeff_abs",*B);
  const Exact c=L* *B; r.candidate_coeff_abs=c;
  if (2*c >= Q) { r.blocker="diagonal_product_centered_lift_unproved"; return; }
  const Exact naive=N* *B + *a * *ideal;
  const Exact error=parent.fresh_slot_error ? std::min(naive,*parent.fresh_slot_error) : naive;
  set(m,"parent_slot_error_abs",error);
  out.fresh_coeff=c; r.bias=C*error; r.blocker.clear();
  r.formula="b2j/same-key-diagonal-product-v1"; // no signed scalar gain
}
} // namespace snabf::ir::noise
