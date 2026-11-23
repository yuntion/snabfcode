package epochs

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const RequireFreshMulRescaleV1 EncodingPolicy = 3

// Unlike Runtime this handle exposes no evaluator, mutable plaintext binding,
// parameters or externally supplied ciphertext. Generated New owns the Program.
type FreshModel struct{ bound *Runtime }

func BindFresh(s Program, evaluate Evaluation, policy EncodingPolicy) (*FreshModel, error) {
	if policy != RequireFreshMulRescaleV1 {
		return nil, fmt.Errorf("missing fresh mul/rescale binding")
	}
	r, err := bind(s, evaluate, true, policy)
	if err != nil {
		return nil, err
	}
	return &FreshModel{bound: r}, nil
}

func freshShape(s Program, p ckks.Parameters) bool {
	if len(s.Epochs) != 1 || s.Switch != nil || s.Source != s.Target || s.Source != p.N() ||
		s.Width < 1 || s.Width > 64 || s.Width&(s.Width-1) != 0 || s.Slots < 16 || s.Slots > 32768 ||
		p.N() < 1024 || p.N() > 65536 || p.RingType() != ring.Standard || p.MaxLevelP() != 0 || p.MaxLevel() > 7 ||
		s.Epochs[0].Scale < 1 || s.Epochs[0].Scale >= 47 || p.EncodingPrecision() != 53 || p.LevelsConsumedPerRescaling() != 1 || p.P()[0] >= 1<<50 ||
		p.Xs() != (ring.Ternary{P: 2.0 / 3.0}) || p.Xe() != (ring.DiscreteGaussian{Sigma: 3.2, Bound: 19.2}) {
		return false
	}
	for _, q := range p.Q() {
		if q >= 1<<50 {
			return false
		}
	}
	return true
}

// This entry couples the domain check to the very EncryptNew used by the
// generated evaluation. A valid scale/shape on an arbitrary ciphertext is
// not evidence of fresh PK provenance. No ciphertext or secret is exported.
func (m *FreshModel) EvalFresh(x []float64) (*rlwe.Ciphertext, error) {
	if m == nil || m.bound == nil {
		return nil, fmt.Errorf("missing fresh mul/rescale binding")
	}
	r := m.bound
	if len(x) != r.Program.Width {
		return nil, fmt.Errorf("wrong logical width")
	}
	input := make([]float64, len(x))
	for i, v := range x {
		if !finite(v) || math.Abs(v) > 1 {
			return nil, fmt.Errorf("outside fresh unit input domain")
		}
		input[i] = v
	}
	ct, err := r.Encrypt(input)
	if err != nil {
		return nil, err
	}
	return r.Eval(ct)
}

// Decode only the output of this guarded call, never a caller's ciphertext.
func (m *FreshModel) EvalFreshDecoded(x []float64) ([]float64, error) {
	ct, err := m.EvalFresh(x)
	if err != nil {
		return nil, err
	}
	return m.bound.Decode(ct)
}
