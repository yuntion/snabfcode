// Planning workload: explicit incoming budget on a real HEIR bootstrapping op.
!bt = !lattigo.ckks.bootstrapping_evaluator
!ct = !lattigo.rlwe.ciphertext
!ev = !lattigo.ckks.evaluator
!p = !lattigo.ckks.parameter
!enc = !lattigo.ckks.encoder
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [288230376150630401, 68190209, 66551809, 67731457, 66813953, 67502081, 67043329, 67239937], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 26>, scheme.ckks} {
  func.func @bootstrap(%boot: !bt, %eval: !ev, %params: !p, %encoder: !enc, %in: !ct {snabf.slot_period = 4096 : i64}) -> !ct {
    %out = lattigo.ckks.bootstrap %boot, %in {snabf_input_level = 3 : i64, snabf_live_slots = 4096 : i64} : (!bt, !ct) -> !ct
    return %out : !ct
  }
}
