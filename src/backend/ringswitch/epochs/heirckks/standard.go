package heirckks
import (
    "github.com/tuneinsight/lattigo/v6/core/rlwe"
    "github.com/tuneinsight/lattigo/v6/schemes/ckks"
)
func Add(evaluator *ckks.Evaluator, ct *rlwe.Ciphertext, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext) {
  ct2, err0 := evaluator.AddNew(ct, ct1)
  if err0 != nil {
    panic(err0)
  }
  return ct2
}
func AddPlain(evaluator *ckks.Evaluator, ct *rlwe.Ciphertext, pt *rlwe.Plaintext) (*rlwe.Ciphertext) {
  ct1, err1 := evaluator.AddNew(ct, pt)
  if err1 != nil {
    panic(err1)
  }
  return ct1
}
func Mul(evaluator *ckks.Evaluator, ct *rlwe.Ciphertext, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext) {
  ct2, err2 := evaluator.MulNew(ct, ct1)
  if err2 != nil {
    panic(err2)
  }
  return ct2
}
func MulPlain(evaluator *ckks.Evaluator, ct *rlwe.Ciphertext, pt *rlwe.Plaintext) (*rlwe.Ciphertext) {
  ct1, err3 := evaluator.MulNew(ct, pt)
  if err3 != nil {
    panic(err3)
  }
  return ct1
}
func Rescale(evaluator *ckks.Evaluator, ct *rlwe.Ciphertext) (*rlwe.Ciphertext) {
  ct1 := ct.CopyNew()
  err4 := evaluator.Rescale(ct, ct1)
  if err4 != nil {
    panic(err4)
  }
  return ct1
}
