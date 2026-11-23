#include "ir/NativeScaleLower.h"
#include "ir/NativePhysical.h"
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

using namespace snabf;
using namespace snabf::native;
namespace {
using R = ExactRational;
using T = ExactScaleTracker;
using Mode = NativeScaleMode;

void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
template<class F> void refused(F f, std::string_view why) {
  try { f(); } catch (const std::invalid_argument &e) {
    check(std::string_view(e.what()).find(why) != std::string_view::npos, e.what());
    return;
  }
  throw std::runtime_error("accepted: " + std::string(why));
}

ScaleDomain domain() { return {"key", 16, 40, {353, 449, 97}, {193, 257}}; }
NativeScaleAddRequest request(ScaleState l, ScaleState h, int level = 1, ScaleDomain d = domain()) {
  NativeScaleAddRequest r{};
  r.result_id = "join";
  r.lhs.value = {"left", d, level, 8, l};
  r.rhs.value = {"right", d, level, 8, h};
  return r;
}
NativeScaleAddPlan ready(const NativeScaleAddRequest &r) {
  return std::get<NativeScaleAddPlan>(lowerNativeScaleAdd(r));
}
EncodingCallContext call(const NativeScaleValue &v, const std::string &site, int precision = 53) {
  return {"encode:" + site, site, 1, v.level, v.slots, v.domain, true, "[]float64", 53, precision, false};
}
void allowExplicit(NativeScaleOperand &s, int precision = 53) {
  auto site = "align:" + s.value.id;
  s.align_one = NativeScaleAlignSite{call(s.value, site, precision), "rescale:" + site, "drop:" + site};
}
void bindSource(NativeScaleOperand &s, const R &requested) {
  const auto &root = s.public_mul ? s.public_mul->input : s.value;
  auto c = call(root, root.id);
  c.dynamic_input = true;
  c.operand_index = 0;
  s.source_encoding = T::encode(c, requested, {});
}
// Construct a real existing definition. Its product/result scale is derived,
// never supplied by the join. Keep a signed zero in the public payload.
void publicMul(NativeScaleAddRequest &r, bool left, ScaleState in, const R &requested,
               int precision = 53, std::vector<double> payload = {2., -0.}) {
  auto &s = left ? r.lhs : r.rhs;
  const auto &v = s.value;
  NativeScaleValue root{"source:" + v.id, v.domain, v.level + 1, v.slots, in};
  auto c = call(root, "mul:" + v.id, precision);
  auto old = T::encode(c, requested, payload);
  std::vector<uint64_t> tail{v.domain.q.at(root.level)};
  auto product = T::multiply(in, {old.realized_meta_scale, old.realized_eff_scale});
  auto rs = T::rescaleForLowering(product, v.domain, root.level, tail, 0);
  s.public_mul = NativeScalePublicMul{root, c, requested, payload, old, tail,
      {{c.site_id, 1}}, {{v.id, 0}}, {{r.result_id, left ? 0 : 1}}};
  s.value.scale = {rs.meta_after, rs.eff_after};
}

void exactAndDomain() {
  auto r = request({8, 8}, {8, 8});
  auto p = ready(r);
  check(p.mode == Mode::None && p.preparation.empty() && p.add.level == 1 &&
        p.add.scales.common_effective_scale == R(8) && p.charged_decoded_bias == 0 &&
        !p.add.scales.decoded_bias && p.add.scales.missing_operand_bounds, "exact uncharged add");
  for (int i = 0; i < 5; ++i) {
    auto bad = r;
    auto &v = bad.rhs.value;
    if (i == 0) std::swap(v.domain.q[0], v.domain.q[1]);
    if (i == 1) std::swap(v.domain.p[0], v.domain.p[1]);
    if (i == 2) v.domain.key_id += "!";
    if (i == 3) --v.level;
    if (i == 4) v.slots /= 2;
    refused([&]{ ready(bad); }, "identical ordered Q/P");
  }
  r.decoded_bias_budget = -1;
  refused([&]{ ready(r); }, "bound or budget");
}

void blockSwitch() {
  ScaleDomain d{"key",16,40,{97,193,257},{449,577}};
  const R factor=R(10)*16*16/(449*577);
  check(nativeBlockSwitchError(d,0)==factor*98,"partial first digit");
  check(nativeBlockSwitchError(d,1)==factor*(97*193+1),"two-limb digit");
  check(nativeBlockSwitchError(d,2)==factor*(97*193+1+258),"partial final digit");
  refused([&]{nativeBlockSwitchError(d,3);},"domain/level");
  d.p.resize(1);
  check(nativeBlockSwitchError(d,2)==R(10)*16*16*(98+194+258)/449,"one-P baseline");
}

void bounded() {
  // One RN128 ULP near 2^40. Both metadata round to the same binary64.
  const R m("1099511627776"), u = R(1) / R(mpz_class(1) << 87), h = m + u;
  check(m.get_d() == h.get_d(), "expected host-double collision");
  auto r = request({m, m-u}, {h, h+u});
  r.lhs.abs_bound = 2;
  r.rhs.abs_bound = 3;
  const R charge = 7*u/h; // |-2u|*2 + |u|*3, in decoded units
  refused([&]{ ready(r); }, "unresolved native add");
  r.decoded_bias_budget = 5*u/h; // either term alone would fit
  refused([&]{ ready(r); }, "unresolved native add");
  r.decoded_bias_budget = charge;
  auto p = ready(r);
  const auto &j = p.add.scales;
  check(p.mode == Mode::BoundedMismatchBias && p.preparation.empty() && !j.common_effective_scale &&
        j.out_meta_scale == h && j.lhs_int_mul == 1 && j.rhs_int_mul == 1 &&
        j.lhs_signed_effect == -2*u && j.rhs_signed_effect == u && j.encoded_bias == R(7*u) &&
        j.decoded_bias == charge && p.charged_decoded_bias == charge &&
        p.add.metadataRatio() == m/h && p.add.effectiveRatio() == (m-u)/(h+u) &&
        p.add.lhs_before == r.lhs.value.scale && p.add.rhs_before == r.rhs.value.scale &&
        r.lhs.value.scale == ScaleState{m, m-u}, "one-ULP two-sided proof or metadata rewrite");
  std::swap(r.lhs, r.rhs);
  auto reverse = ready(r);
  check(reverse.add.scales.lhs_signed_effect == u && reverse.add.scales.rhs_signed_effect == -2*u &&
        reverse.charged_decoded_bias == charge, "swapped operands must swap coefficients and bounds");

  const R next(std::nextafter(1., 2.)); // one binary64 ULP
  r = request({1, 1}, {next, next});
  r.lhs.abs_bound = 1;
  r.decoded_bias_budget = (next - 1)/next;
  refused([&]{ ready(r); }, "unresolved native add"); // even the zero-residual side needs a bound
  r.rhs.abs_bound = 0;
  check(ready(r).charged_decoded_bias == *r.decoded_bias_budget, "binary64 ULP was ignored");
  r.lhs.abs_bound = 0;
  r.decoded_bias_budget.reset();
  check(ready(r).charged_decoded_bias == 0, "known zero is not an unknown magnitude");

  r = request({100000000, 100000000}, {100000001, 100000001});
  r.lhs.abs_bound = r.rhs.abs_bound = r.decoded_bias_budget = 1;
  check(ready(r).mode == Mode::BoundedMismatchBias, "inclusive exact pre-filter");
  r.rhs.value.scale = {100000002, 100000002};
  refused([&]{ ready(r); }, "unresolved native add"); // a large allocation does not relax the filter
  r = request({8, 7}, {8, 7});
  refused([&]{ ready(r); }, "unresolved native add"); // common E is not necessarily M
}

void folds() {
  for (bool left : {true, false}) {
    auto r = request({12, 12}, {12, 12});
    publicMul(r, left, {8, 8}, 97);
    auto &s = left ? r.lhs : r.rhs;
    const auto &f = *s.public_mul;
    bindSource(s, 8);
    bindSource(left ? r.rhs : r.lhs, 12);
    allowExplicit(s);
    auto result = lowerNativeScaleAdd(r, [](const auto &) -> std::optional<ScaleDomain> {
      throw std::runtime_error("solver called before eligible fold");
    });
    const auto &p = std::get<NativeScaleAddPlan>(result);
    check(p.mode == Mode::FoldIntoPlain && p.preparation.size() == 3 && p.fold &&
          p.fold->rho == R(3,2) && p.fold->exact_representable && p.fold->residual_scale_diff == 0 &&
          p.mul_depth_delta == 0 && p.levels_consumed == 0 && p.dropped_q.empty(), "fold cost or ratio");
    const auto &pt = std::get<NativeEncodePlain>(p.preparation[0]);
    const auto &mul = std::get<NativeMulPlain>(p.preparation[1]);
    const auto &rs = std::get<NativeRescaleOne>(p.preparation[2]);
    T::verifyEncoding(pt.encoding, f.call, R(291,2), f.logical_payload);
    check(pt.encoding.branch == "embed_double_f64" &&
          pt.encoding.argument_sha256 == f.original_encoding.argument_sha256 && pt.payload.size() == 2 &&
          pt.payload[0] == 2 && std::bit_cast<uint64_t>(pt.payload[1]) == std::bit_cast<uint64_t>(-0.) &&
          mul.ciphertext == f.input.id && mul.plaintext == f.call.call_id && mul.result == rs.input &&
          rs.result == s.value.id && rs.step.levels_consumed == 1 && rs.step.mul_depth_delta == 0 &&
          rs.step.meta_after == 12 && rs.step.eff_after == 12 && p.add.level == 1 &&
          p.add.lhs == r.lhs.value.id && p.add.rhs == r.rhs.value.id && s.value.scale == ScaleState{8,8},
          "fold changed a weight, SSA edge, logical level or original binding");
  }

  auto r = request({1,1}, {12,12});
  publicMul(r, true, {8,8}, 97);
  for (int i = 0; i != 4; ++i) {
    auto bad = r;
    auto &f = *bad.lhs.public_mul;
    if (i == 0) f.plain_uses.push_back({"side",1});
    if (i == 1) f.product_uses.push_back({"side",0});
    if (i == 2) f.rescale_uses.push_back({"return",0});
    if (i == 3) f.rescale_uses[0].operand_index = 1;
    refused([&]{ ready(bad); }, "unresolved native add");
  }
  auto bad = r;
  bad.lhs.public_mul->logical_payload[1] = 0.;
  refused([&]{ ready(bad); }, "stale encoding");
  bad = r;
  bad.lhs.value.scale.effective = 7;
  refused([&]{ ready(bad); }, "stale existing");
  bad = r;
  bad.lhs.public_mul->dropped_q = {449};
  refused([&]{ ready(bad); }, "ordered Q suffix");
  bad = r;
  bad.lhs.public_mul->dropped_q = {97,449};
  refused([&]{ ready(bad); }, "UNSUPPORTED_MULTIPRIME_RESCALE");
  bad = r;
  bad.rhs.value.scale = {24,24}; // rho = 3, outside the search window
  refused([&]{ ready(bad); }, "unresolved native add");

  r = request({1,1}, {1,1});
  publicMul(r, true, {3,3}, 32);
  r.lhs.abs_bound = 2;
  r.rhs.abs_bound = 7;
  refused([&]{ ready(r); }, "unresolved native add");
  const R realized(97./3.);
  const R residual = 3*realized/97 - 1;
  r.decoded_bias_budget = 2*abs(residual);
  auto p = ready(r);
  check(p.mode == Mode::FoldIntoPlain && !p.fold->exact_representable &&
        p.fold->requested_pt_scale == R(97,3) && p.fold->realized_pt_meta != realized &&
        p.fold->realized_pt_eff == realized && residual != 0 && p.fold->residual_scale_diff == residual &&
        p.charged_decoded_bias == *r.decoded_bias_budget && !p.add.scales.common_effective_scale,
        "rounded fold lost its requested/realized scales or uncharged actual bias");
}

void explicitAlignment() {
  for (bool left : {true, false}) {
    auto r = request({12,12}, {12,12}, 2);
    auto &s = left ? r.lhs : r.rhs, &peer = left ? r.rhs : r.lhs;
    s.value.scale = {8,8};
    allowExplicit(s);
    r.min_output_level = 1;
    auto p = ready(r);
    check(p.mode == Mode::ExplicitAlignMul && p.preparation.size() == 4 && !p.fold &&
          p.mul_depth_delta == 1 && p.levels_consumed == 1 && p.dropped_q == std::vector<uint64_t>{97},
          "explicit alignment cost");
    const auto &pt = std::get<NativeEncodePlain>(p.preparation[0]);
    const auto &mul = std::get<NativeMulPlain>(p.preparation[1]);
    const auto &rs = std::get<NativeRescaleOne>(p.preparation[2]);
    const auto &drop = std::get<NativeDropLevel>(p.preparation[3]);
    T::verifyEncoding(pt.encoding, s.align_one->call, R(291,2), std::array{1.});
    check(pt.payload == std::vector<double>{1.} && pt.encoding.argument_sha256 &&
          mul.plaintext == pt.encoding.call_id && mul.ciphertext == s.value.id &&
          mul.result == rs.input && mul.after == ScaleState{1164,1164} &&
          rs.step.in_level == 2 && rs.step.out_level == 1 && rs.step.mul_depth_delta == 1 &&
          rs.step.levels_consumed == 1 && rs.step.divisions == std::vector<PrimeDivision>{{97,{12,12}}} &&
          drop.input == peer.value.id && drop.from_level == 2 && drop.to_level == 1 &&
          drop.unchanged == peer.value.scale && p.add.level == drop.to_level &&
          p.add.lhs == (left ? rs.result : drop.result) && p.add.rhs == (left ? drop.result : rs.result) &&
          p.add.scales.out_meta_scale == 12 && p.add.scales.common_effective_scale == R(12) &&
          s.value.scale == ScaleState{8,8} && peer.value.level == 2, "wrong explicit physical instruction");
    r.min_output_level = 2;
    refused([&]{ ready(r); }, "unresolved native add");
    r.min_output_level = 0;
    s.align_one->peer_drop_id = peer.value.id;
    refused([&]{ ready(r); }, "fresh SSA");
  }
  // A metadata-only comparison would 'repair' this on the f64 branch.
  const R odd("9007199254740993");
  auto r = request({97,97}, {odd,odd}, 2);
  allowExplicit(r.lhs, 53);
  refused([&]{ ready(r); }, "unresolved native add");
  allowExplicit(r.lhs, 60);
  auto p = ready(r);
  const auto &pt = std::get<NativeEncodePlain>(p.preparation[0]);
  check(pt.encoding.branch == "bigfloat_value" && pt.encoding.requested_scale == odd &&
        pt.encoding.realized_eff_scale == odd && p.add.scales.common_effective_scale == odd,
        "explicit encode branch was guessed or its request rounded through f64");
  auto d = domain();
  d.default_log_scale = 80;
  auto two = request({8,8}, {12,12}, 2, d);
  allowExplicit(two.lhs);
  refused([&]{ ready(two); }, "UNSUPPORTED_MULTIPRIME_RESCALE");
  two.rhs.value.scale = two.lhs.value.scale;
  refused([&]{ ready(two); }, "UNSUPPORTED_MULTIPRIME_RESCALE");
}

NativeScaleAddRequest strictInput(const ScaleDomain &d, std::vector<double> weights = {2., -0.}) {
  auto r = request({1,1}, {8,8}, 1, d);
  // The exact scale equation is 8*449/Q[2] = 8. Q[2] = 97 in the
  // original graph, so folding would be outside its search window.
  publicMul(r, true, {8,8}, 449, 53, std::move(weights));
  bindSource(r.lhs, 8);
  bindSource(r.rhs, 8);
  return r;
}

void strictResolve() {
  auto r = strictInput(domain());
  auto candidate = domain();
  std::swap(candidate.q[1], candidate.q[2]);
  int solves = 0;
  auto solver = [&](const NativeScaleAddRequest &live) -> std::optional<ScaleDomain> {
    ++solves;
    const auto &f = *live.lhs.public_mul;
    const R required = f.input.scale.effective * f.original_encoding.realized_eff_scale /
                       live.rhs.value.scale.effective;
    for (const auto &d : {domain(), candidate})
      if (R(std::to_string(d.q.at(f.input.level))) == required) return d;
    return std::nullopt;
  };
  auto result = lowerNativeScaleAdd(r, solver);
  const auto &restart = std::get<NativeStrictScaleRestart>(result);
  check(solves == 1 && restart.requiredDomain() == candidate, "strict solution must restart the graph");
  refused([&]{ restart.finish(r); }, "rebuilt join");
  auto rebuilt = strictInput(candidate);
  auto p = restart.finish(rebuilt);
  check(p.mode == Mode::ExactPrimeMatch && p.domain == candidate && p.preparation.empty() &&
        p.add.level == 1 && p.add.scales.common_effective_scale == R(8) && p.charged_decoded_bias == 0,
        "re-solved physical join");
  check(std::get<NativeScaleAddPlan>(lowerNativeScaleAdd(rebuilt, solver)).mode == Mode::None && solves == 1,
        "an already equal pair was reported as a strict re-solve");

  auto bad = rebuilt;
  std::swap(bad.lhs.value.domain.p[0], bad.lhs.value.domain.p[1]);
  bad.rhs.value.domain = bad.lhs.value.domain;
  refused([&]{ restart.finish(bad); }, "rebuilt join");
  bad = rebuilt;
  bad.lhs.value.scale.effective += R(1) / R(mpz_class(1) << 127);
  bad.lhs.abs_bound = bad.rhs.abs_bound = bad.decoded_bias_budget = 1;
  refused([&]{ restart.finish(bad); }, "exact metadata and effective");
  bad = rebuilt;
  bad.lhs.value.scale = bad.rhs.value.scale = {9,9}; // forged exact match, no producing instructions
  refused([&]{ restart.finish(bad); }, "stale existing");
  bad = strictInput(candidate, {3., -0.}); // internally consistent, different source weight
  refused([&]{ restart.finish(bad); }, "changed an encoding");
  bad = rebuilt;
  bad.rhs.source_encoding.reset();
  refused([&]{ restart.finish(bad); }, "source-bound producer");
  bad = rebuilt;
  bad.lhs.value.id += "!";
  refused([&]{ restart.finish(bad); }, "rebuilt join");

  refused([&]{ lowerNativeScaleAdd(r, [](const auto &) { return domain(); }); }, "new ordered Q");
  auto wrong = domain();
  std::swap(wrong.q[0], wrong.q[1]); // valid new basis, same divisor: not a solution
  refused([&]{ lowerNativeScaleAdd(r, [&](const auto &) { return wrong; }); }, "concrete scale equation");
  r.rhs.source_encoding.reset();
  refused([&]{ lowerNativeScaleAdd(r, solver); }, "unresolved native add");
  check(solves == 1, "unreplayable producers must not authorize Tier 2");
}
} // namespace

int main() {
  try {
    exactAndDomain();
    blockSwitch();
    bounded();
    folds();
    strictResolve();
    explicitAlignment();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
