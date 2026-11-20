!bootstrapping_eval_keys = !lattigo.ckks.bootstrapping_eval_keys
!bootstrapping_evaluator = !lattigo.ckks.bootstrapping_evaluator
!bootstrapping_param = !lattigo.ckks.bootstrapping_parameter
!ct = !lattigo.rlwe.ciphertext
!decryptor = !lattigo.rlwe.decryptor
!encoder = !lattigo.ckks.encoder
!encryptor_pk = !lattigo.rlwe.encryptor<publicKey = true>
!evaluator = !lattigo.ckks.evaluator
!kgen = !lattigo.rlwe.key_generator
!param = !lattigo.ckks.parameter
!pk = !lattigo.rlwe.public_key
!sk = !lattigo.rlwe.secret_key
module attributes {scheme.ckks} {
  func.func @bootstrap(%bootstrapping_evaluator: !bootstrapping_evaluator, %evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct) -> !ct {
    %ct_0 = lattigo.ckks.bootstrap %bootstrapping_evaluator, %ct {snabf.boot_costs = {boot_costs = [{cost = 8.000000e+01 : f64, level = 3 : i64, n = 16384 : i64}, {cost = 4.000000e+01 : f64, level = 2 : i64, n = 8192 : i64}], down_cost = 1.000000e+01 : f64, downstream_bins = array<f64: 5.000000e+00, 5.000000e+00, 2.000000e+01, 1.500000e+01>, gadget_weights = array<f64: 1.000000e+00, 2.000000e+00>, key_variance = 7.32421875E-4 : f64, layout_verified = true, multiplicity = 1 : i64, target_n = 8192 : i64, up_cost = 2.000000e+01 : f64}, snabf.ring_switch = {enabled = true, margin = 5.000000e+00 : f64, penalty = 3.500000e+01 : f64, reason = "profitable", rs_variance = 5.000000e+00 : f64, savings = 4.000000e+01 : f64, small_level = 2 : i64}, snabf_input_level = 3 : i64, snabf_live_slots = 4096 : i64} : (!bootstrapping_evaluator, !ct) -> !ct
    return %ct_0 : !ct
  }
  func.func @bootstrap__configure() -> (!bootstrapping_evaluator, !evaluator, !param, !encoder, !encryptor_pk, !decryptor) {
    %param = lattigo.ckks.new_parameters_from_literal  <paramsLiteral = <logN = 14, Q = [288230376150630401, 68190209, 66551809, 67731457, 66813953, 67502081, 67043329, 67239937], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 26>> : () -> !param
    %encoder = lattigo.ckks.new_encoder %param : (!param) -> !encoder
    %kgen = lattigo.rlwe.new_key_generator %param : (!param) -> !kgen
    %secretKey, %publicKey = lattigo.rlwe.gen_key_pair %kgen : (!kgen) -> (!sk, !pk)
    %encryptor = lattigo.rlwe.new_encryptor %param, %publicKey : (!param, !pk) -> !encryptor_pk
    %decryptor = lattigo.rlwe.new_decryptor %param, %secretKey : (!param, !sk) -> !decryptor
    %bootstrapping_param = lattigo.ckks.new_bootstrapping_parameters_from_literal %param <btParamsLiteral = <logN = 14>> : (!param) -> !bootstrapping_param
    %bootstrapping_eval_keys = lattigo.ckks.gen_evaluation_keys_bootstrapping %bootstrapping_param, %secretKey : (!bootstrapping_param, !sk) -> !bootstrapping_eval_keys
    %bootstrapping_evaluator = lattigo.ckks.new_bootstrapping_evaluator %bootstrapping_param, %bootstrapping_eval_keys : (!bootstrapping_param, !bootstrapping_eval_keys) -> !bootstrapping_evaluator
    %evaluator = lattigo.ckks.new_evaluator %param : (!param) -> !evaluator
    return %bootstrapping_evaluator, %evaluator, %param, %encoder, %encryptor, %decryptor : !bootstrapping_evaluator, !evaluator, !param, !encoder, !encryptor_pk, !decryptor
  }
}

