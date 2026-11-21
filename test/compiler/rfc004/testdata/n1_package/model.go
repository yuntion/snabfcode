// Package mechanics fixture, not a compiled N1 graph or a security estimate.
package planned

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const PlanSHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

// Both consumers need the original degree-2 product, not each other's output buffer.
func evaluate(e *ckks.Evaluator, x *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	product, err := e.MulNew(x, x)
	if err != nil {
		return nil, err
	}
	left, err := relin(e, product)
	if err != nil {
		return nil, err
	}
	right, err := relin(e, product)
	if err != nil {
		return nil, err
	}
	return e.AddNew(left, right)
}
