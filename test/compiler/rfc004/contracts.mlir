// RFC-004 N0 inspection fixture. These are not executable HE parameters.
#in = #snabf.layout_spec<{kind="packed_multi", logical_dims=array<i64: 2,3>,
  axis_order=array<i64: 0,1>, slot_map=array<i64: 0,1,2,3,4,5>,
  used_slots=6:i64, period=8:i64, num_slots=8:i64, rotate_direction="left"}>
#out = #snabf.layout_spec<{kind="packed_multi", logical_dims=array<i64: 2>,
  axis_order=array<i64: 0>, slot_map=array<i64: 0,3>,
  used_slots=2:i64, period=8:i64, num_slots=8:i64, rotate_direction="left"}>
#src = #snabf.frontend_provenance<{mode="native-linalg", scale_policy="narrow",
  source_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  inputs=[{func_symbol=@compute, arg_index=0:i64, source_id="x", logical_type=tensor<2x3xf32>, layout=#in, input_domain=unit},
          {func_symbol=@compute, arg_index=1:i64, source_id="y", logical_type=tensor<2x3xf32>, layout=#in}],
  outputs=[{func_symbol=@compute, result_index=0:i64, logical_type=tensor<2xf32>, layout=#out}]}>
#eight = #snabf.exact_scale<"8","1">
#four = #snabf.exact_scale<"4","1">
#enc_add = #snabf.encoding_call_receipt<{call_id="encode-add", site="add", operand_index=1:i64,
  batched=true, arg_type="[]float64", arg_precision=53:i64, encoder_precision=53:i64,
  branch="embed_double_f64", quantization="batched-f64", payload_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  requested=#eight, metadata=#eight, effective=#eight}>
#enc_mul = #snabf.encoding_call_receipt<{call_id="encode-mul", site="mul", operand_index=1:i64,
  batched=true, arg_type="[]float64", arg_precision=53:i64, encoder_precision=53:i64,
  branch="embed_double_f64", quantization="batched-f64", payload_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  requested=#eight, metadata=#eight, effective=#eight}>
#enc = #snabf.encoding_call_receipt<{call_id="encode-constant", site="align", operand_index=1:i64,
  batched=true, arg_type="[]float64", arg_precision=53:i64, encoder_precision=53:i64,
  branch="embed_double_f64", quantization="batched-f64",
  payload_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  requested=#snabf.exact_scale<"449","1">, metadata=#snabf.exact_scale<"449","1">, effective=#snabf.exact_scale<"449","1">}>
#plan = #snabf.scale_plan<{mode="explicit_align_mul", plaintext="encoded-one", producer="x", consumer="join", operand_index=0:i64,
  ordered_q=array<i64: 97,353,449>, ordered_p=array<i64: 193>, level_in=2:i64, level_out=1:i64,
  in_scale=#eight, out_scale=#eight, mul_depth_delta=1:i64, levels_consumed=1:i64, dropped_q=array<i64: 449>}>
#exact = #snabf.scale_plan<{mode="exact_prime_match", producer="x", consumer="sub", operand_index=0:i64,
  ordered_q=array<i64: 97,353,449>, ordered_p=array<i64: 193>, level_in=2:i64, level_out=2:i64,
  in_scale=#eight, out_scale=#eight, mul_depth_delta=0:i64, levels_consumed=0:i64, dropped_q=array<i64>}>
!x = !snabf.ct<tensor<2x3xf32>,16,2,@key>
!low = !snabf.ct<tensor<2x3xf32>,16,1,@key>
!y = !snabf.ct<tensor<2xf32>,16,2,@key>
module attributes {snabf.frontend_provenance=#src} {
  "snabf.profile"() {sym_name="key", ring=16:i64, slots=8:i64, q=array<i64: 97,353,449>, p=array<i64: 193>,
    default_log_scale=3:i64, key_id="test", layout="packed", key_bytes=1:i64} : () -> ()
  func.func @compute(%x: !x, %z: !x) -> !y {
    %pt = arith.constant dense<1.0> : tensor<2x3xf32>
    %table = arith.constant dense<1.0> : tensor<2x6xf32>
    %init = arith.constant dense<[7.0, 11.0]> : tensor<2xf32>
    %0 = "snabf.ct_mul"(%x,%z) {lhs_scale=#eight, rhs_scale=#four, requested_product=#snabf.exact_scale<"32","1">} : (!x,!x) -> !x
    %1 = "snabf.sub"(%x,%z) {scale_plan=#exact} : (!x,!x) -> !x
    %2 = "snabf.negate"(%x) : (!x) -> !x
    %3 = "snabf.add_plain"(%x,%pt) {site="add", encoding=#enc_add} : (!x,tensor<2x3xf32>) -> !x
    %4 = "snabf.mul_plain"(%x,%pt) {site="mul", encoding=#enc_mul} : (!x,tensor<2x3xf32>) -> !x
    %5 = "snabf.align_scale"(%x) {site="align", scale_plan=#plan, encoding=#enc} : (!x) -> !low
    %6 = "snabf.rotate"(%x) {offset=1:i64, layout=#in} : (!x) -> !x
    %7 = "snabf.lintrans"(%x,%table,%init) {diagonals=array<i64: 0,1>, in_layout=#in, out_layout=#out} : (!x,tensor<2x6xf32>,tensor<2xf32>) -> !y
    %8 = "snabf.reduce_sum"(%x) {in_layout=#in, out_layout=#out, axes=array<i64: 1>, keep_dims=false,
      groups=[array<i64: 0,1,2>,array<i64: 3,4,5>], valid_slots=array<i64: 0,1,2,3,4,5>,
      output_slots=array<i64: 0,3>, edge_divisors=array<i64: 3,3>} : (!x) -> !y
    return %8 : !y
  }
}
