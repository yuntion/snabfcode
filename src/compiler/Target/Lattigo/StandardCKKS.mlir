// Ring, keys and scales belong to the calling evaluator and operands. This
// file contains only ordinary HEIR operators; it knows nothing about SN-ABF.
!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext
!ev = !lattigo.ckks.evaluator
module attributes {scheme.ckks} {
  func.func @add(%e: !ev, %a: !ct, %b: !ct) -> !ct {
    %r = lattigo.ckks.add_new %e, %a, %b : (!ev, !ct, !ct) -> !ct
    return %r : !ct
  }
  func.func @addPlain(%e: !ev, %a: !ct, %b: !pt) -> !ct {
    %r = lattigo.ckks.add_new %e, %a, %b : (!ev, !ct, !pt) -> !ct
    return %r : !ct
  }
  func.func @mul(%e: !ev, %a: !ct, %b: !ct) -> !ct {
    %r = lattigo.ckks.mul_new %e, %a, %b : (!ev, !ct, !ct) -> !ct
    return %r : !ct
  }
  func.func @mulPlain(%e: !ev, %a: !ct, %b: !pt) -> !ct {
    %r = lattigo.ckks.mul_new %e, %a, %b : (!ev, !ct, !pt) -> !ct
    return %r : !ct
  }
  func.func @rescale(%e: !ev, %a: !ct) -> !ct {
    %r = lattigo.ckks.rescale_new %e, %a : (!ev, !ct) -> !ct
    return %r : !ct
  }
}
