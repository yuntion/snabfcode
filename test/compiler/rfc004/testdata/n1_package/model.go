// Package mechanics fixture, not a compiled N1 graph or a security estimate.
package planned

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	epoch "snabf.local/compiler/src/backend/ringswitch/epochs"
)

const PlanSHA256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

// Test-only binding for the external observation driver (no admission or security claim).
func New() (*epoch.Runtime, error) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 10, LogQ: []int{46, 40, 40}, LogP: []int{45}, LogDefaultScale: 35})
	if err != nil {
		return nil, err
	}
	s := epoch.Program{Fingerprint: PlanSHA256, Source: p.N(), Target: p.N(), Width: 1, Slots: 512,
		Epochs: []epoch.Epoch{{N: p.N(), Q: p.Q(), P: p.P(), Scale: 35, KeyID: "test/source", Relin: true}}}
	return epoch.Bind(s, func(r *epoch.Runtime, x *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		return evaluate(r.Evaluator(p.N()), x)
	})
}

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
