// Separate periodic-encoding workload. Calibrations are regression cost units.
!bt = !lattigo.ckks.bootstrapping_evaluator
!ct = !lattigo.rlwe.ciphertext
!ev = !lattigo.ckks.evaluator
!p = !lattigo.ckks.parameter
!enc = !lattigo.ckks.encoder
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1099512938497,2148794369,2150760449,2150891521,2151677953], P = [72057594038321153,72057594040680449], logDefaultScale = 31>, scheme.ckks} {
  func.func @bootstrap(%boot: !bt, %eval: !ev, %params: !p, %encoder: !enc, %in: !ct {snabf.slot_period = 64 : i64}) -> !ct {
    %out = lattigo.ckks.bootstrap %boot, %in {snabf_input_level = 1 : i64, snabf_live_slots = 64 : i64,
      snabf.bootstrap_kernel = "periodic-n15-v1", snabf.boot_costs = {target_n = 32768 : i64, multiplicity = 1 : i64,
      layout_verified = true, layout_contract = "trace-periodic", layout_cost = 0.0 : f64,
      key_variance = 0.00018310546875 : f64, gadget_weights = array<f64: 1.0, 2.0>, downstream_bins = array<f64: 5.0, 5.0>,
      down_cost = 10.0 : f64, up_cost = 20.0 : f64, boot_costs = [{n = 65536 : i64, level = 1 : i64, cost = 80.0 : f64},
      {n = 32768 : i64, level = 0 : i64, cost = 40.0 : f64}]}} : (!bt, !ct) -> !ct
    return %out : !ct
  }
}
