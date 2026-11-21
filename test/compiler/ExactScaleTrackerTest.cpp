#include "ir/ExactScaleTracker.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <limits>

using namespace snabf;
using T = ExactScaleTracker;
using R = ExactRational;
namespace {
void check(bool x, const char *why) { if (!x) throw std::runtime_error(why); }
template<class F> void denied(F f, const char *why) {
  try { f(); } catch (const std::invalid_argument &e) {
    check(std::string(e.what()).find(why) != std::string::npos, e.what()); return;
  }
  throw std::runtime_error(std::string("accepted: ") + why);
}
llvm::json::Object probe() {
  ScaleDomain d{"key", 16, 40, {353,449,97}, {193}};
  EncodingCallContext c{"encode:pt", "mul", 1, 2, 8, d, true, "[]float64", 53, 53, false};
  std::array values{1., -0.};
  const R m("9007199254740993");
  auto enc = T::encode(c, m, values);
  T::verifyEncoding(enc, c, m, values);
  auto hi = c; hi.encoder_prec_bits = 60;
  auto big = T::encode(hi, m, values);
  check(enc.realized_eff_scale == m-1 && big.realized_eff_scale == m, "encoder precision");
  denied([&]{ T::verifyEncoding(enc, hi, m, values); }, "receipt");
  for (int i=0; i<10; ++i) {
    auto bad = enc;
    switch (i) {
    case 0: bad.call_id += "!"; break;
    case 1: ++bad.operand_index; break;
    case 2: bad.site_id += "!"; break;
    case 3: bad.level--; break;
    case 4: bad.quantization_rule = big.quantization_rule; break;
    case 5: bad.branch = big.branch; break;
    case 6: std::swap(bad.domain.q[0], bad.domain.q[1]); break;
    case 7: bad.argument_sha256.reset(); break;
    case 8: bad.requested_scale += 1; break;
    case 9: bad.domain.key_id += "!"; break;
    }
    denied([&]{ T::verifyEncoding(bad, c, m, values); }, "receipt");
  }
  auto coeff = c; coeff.is_batched = false;
  denied([&]{ T::encode(coeff, m, values); }, "branch");
  coeff = c; coeff.arg_type = "[]*big.Float";
  denied([&]{ T::encode(coeff, m, values); }, "branch");
  auto input = c; input.dynamic_input = true;
  check(!T::encode(input, m, {}).argument_sha256, "private payload claimed");
  denied([&]{ T::encode(input, m, values); }, "payload");
  denied([&]{ T::verifyEncoding(enc, c, m, std::array{1., 0.}); }, "receipt");
  ScaleState a{R("1099511627776"),R("1099511627776")}, b{a.metadata+1,a.effective+1};
  auto add = T::analyzeAddSubMismatch(JoinOp::Add, a,b, 1,0);
  auto sub = T::analyzeAddSubMismatch(JoinOp::Sub, b,a, 0,1);
  check(add.lhs_signed_effect == -1 && sub.rhs_signed_effect == 1 && !sub.common_effective_scale &&
        add.encoded_bias == R(1) && *add.decoded_bias == R(1)/b.metadata && sub.decoded_bias == add.decoded_bias,
        "near join effects/units");
  auto both = T::analyzeAddSubMismatch(JoinOp::Sub,{4,3},{7,6},2,3);
  check(both.lhs_int_mul==1 && both.rhs_int_mul==1 && both.lhs_signed_effect==-4 && both.rhs_signed_effect==1 &&
        both.encoded_bias==R(11) && both.decoded_bias==R(11,7), "two distinct operand gains");
  check(R(*both.decoded_bias_upper_mpfr) >= *both.decoded_bias &&
        R(std::nextafter(*both.decoded_bias_upper_mpfr, 0.)) < *both.decoded_bias, "MPFR upper endpoint");
  for (bool lhs : {false,true}) {
    std::optional<R> zero(0), absent;
    auto x = T::analyzeAddSubMismatch(JoinOp::Add, a,a, lhs ? absent : zero, lhs ? zero : absent);
    check(x.missing_operand_bounds && !x.encoded_bias && !x.decoded_bias_upper_mpfr, "missing operand became zero");
  }
  auto nine = T::analyzeAddSubMismatch(JoinOp::Add,{4,4},{9,9},1,0);
  check(nine.lhs_int_mul==2 && nine.lhs_signed_effect==-1, "9/4 truncation");
  denied([&]{ T::analyzeAddSubMismatch(JoinOp::Add,a,b,-1,1); }, "magnitude");
  auto h = d; h.q={97,353,449}; h.default_log_scale=80;
  const std::array<uint64_t,2> tail{449,353};
  const std::array<uint64_t,1> q{97};
  auto rs = T::replayRescale({65536,65536},h,2,tail);
  denied([&]{ T::replayPrimeDivision(a,std::array<uint64_t,2>{97,449*353},1); }, "invalid prime");
  denied([&]{ T::replayPrimeDivision(a,h.q,0); }, "level");
  const R single = pass2::roundCKKSScale(R(65536)/449/353);
  T::verifyReplayedRescale(rs,{65536,65536},h,2,tail);
  auto oneShot=rs; oneShot.meta_after=oneShot.divisions.back().after.metadata=single;
  denied([&]{ T::verifyReplayedRescale(oneShot,{65536,65536},h,2,tail); }, "stale rescale");
  auto wrongContext=rs; wrongContext.domain.key_id="different-key";
  denied([&]{ T::verifyReplayedRescale(wrongContext,{65536,65536},h,2,tail); }, "stale rescale");
  check(rs.meta_after != single && rs.eff_after == R(65536)/449/353 && rs.divisions.size()==2 &&
        rs.divisions[0].after.metadata==pass2::roundCKKSScale(R(65536)/449) &&
        rs.mul_depth_delta==0 && rs.levels_consumed==2 && rs.out_level==0, "stepwise rescale");
  denied([&]{ T::replayRescale({65536,65536},h,2,std::array<uint64_t,2>{353,449}); }, "ordered");
  denied([&]{ T::rescaleForLowering(a,h,2,tail,1); }, "UNSUPPORTED_MULTIPRIME_RESCALE");
  denied([&]{ T::rescaleForLowering(a,h,2,q,1); }, "UNSUPPORTED_MULTIPRIME_RESCALE");
  check(T::rescaleForLowering(a,d,2,q,1).mul_depth_delta==1, "explicit alignment depth");
  auto f = T::computeZeroDepthPlainFold({3,3},{1,1},c,values,32,q);
  auto exact = T::computeZeroDepthPlainFold({4,4},{4,4},c,values,97,q);
  check(f.within_search_window && !f.exact_representable && f.rho==R(97,96) && f.residual_scale_diff!=0 &&
        exact.exact_representable && exact.residual_scale_diff==0 &&
        f.replacement.payload_sha256==enc.payload_sha256, "fold representability/payload");
  check(!T::computeZeroDepthPlainFold({3,3},{1,1},c,values,1,q).within_search_window, "fold window");
  auto product = T::multiply({m,m-1},{m,m-1});
  check(product.effective==(m-1)*(m-1) && product.metadata==pass2::roundCKKSScale(m*m), "E was rounded to M");
  // The Python gate compares these predictions with a fresh unmodified Lattigo N0 probe.
  return llvm::json::Object{{"meta",enc.realized_meta_scale.get_str()}, {"float64",enc.realized_eff_scale.get_str()},
          {"batch60",big.realized_eff_scale.get_str()}, {"near_meta",add.out_meta_scale.get_str()},
          {"near_add_coeff",R(add.out_meta_scale+add.lhs_signed_effect).get_str()},
          {"near_sub_coeff",R(-sub.out_meta_scale+sub.rhs_signed_effect).get_str()},
          {"decoded_bias",add.decoded_bias->get_str()}, {"alignment_trunc",std::to_string(both.lhs_int_mul)},
          {"four_nine_coeff",R(nine.out_meta_scale+nine.lhs_signed_effect).get_str()},
          {"after",rs.meta_after.get_str()}, {"after_level",rs.out_level}, {"one_div",single.get_str()},
          {"fold_requested",f.requested_pt_scale.get_str()}, {"fold_actual_meta",f.realized_pt_meta.get_str()},
          {"fold_f64_eff",f.realized_pt_eff.get_str()}, {"fold_residual",f.residual_scale_diff.get_str()}};
}
} // namespace
int main() {
  try { llvm::outs() << llvm::json::Value(probe()) << '\n'; }
  catch (const std::exception &e) { llvm::errs() << e.what() << '\n'; return 1; }
}
