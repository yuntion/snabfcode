// Package ringswitch binds HEIR's bootstrap evaluator to an explicit
// down/refresh/up program on a checked periodic layout.
package ringswitch

import (
	"fmt"
	"math"
	"math/bits"
	"slices"
	"time"

	boot "github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend"
)

type ParametersLiteral = boot.ParametersLiteral
type Plan struct {
	Source, Target, Period, Level, LogScale int
	Q, P                                    []uint64
}
type Parameters struct {
	source, small ckks.Parameters
	circuit       boot.Parameters
	plan          Plan
}
type Keys struct {
	down, up *rlwe.EvaluationKey
	refresh  *boot.EvaluationKeys
}
type Observation struct {
	Rings, Levels [4]int
	LogScales     [4]float64
	SegmentNS     [3]int64
}
type Evaluator struct {
	p       Parameters
	down    *backend.SparseDownConverter
	up      *rlwe.Evaluator
	upKey   *rlwe.EvaluationKey
	refresh *boot.Evaluator
	Last    Observation
}

func NewParametersFromLiteral(src ckks.Parameters, original ParametersLiteral, p Plan) (out Parameters, err error) {
	if original.LogN == nil || *original.LogN != src.LogN() || src.N() != p.Source || p.Target != 1<<15 || p.Source != 2*p.Target ||
		p.Period < 8 || p.Period > p.Target/2 || p.Period&(p.Period-1) != 0 || p.Level < 1 || p.Level > src.MaxLevel() ||
		!slices.Equal(p.Q, src.Q()) || !slices.Equal(p.P, src.P()) || src.LogDefaultScale() != p.LogScale || p.LogScale != 31 {
		return out, fmt.Errorf("checked ring-bootstrap binding disagrees with source profile/layout")
	}
	preset := boot.N15QP880H16384H32
	// The actual residual Q/P are shared across the conversion and must be
	// NTT-compatible in the *large* ring too. Use the circuit factorization
	// from the preset, not a bootstrap secretly executed on the large ring.
	lit := preset.SchemeParams
	lit.Q, lit.P = p.Q, p.P
	lit.LogQ, lit.LogP = nil, nil
	small, err := ckks.NewParametersFromLiteral(lit)
	if err != nil {
		return out, err
	}
	slots := bits.TrailingZeros(uint(p.Period))
	bl := preset.BootstrappingParams
	bl.LogSlots = &slots
	bp, err := boot.NewParametersFromLiteral(small, bl)
	if err != nil {
		return out, err
	}
	if bp.BootstrappingParameters.N() != p.Target {
		return out, fmt.Errorf("wrong refresh ring")
	}
	return Parameters{src, small, bp, p}, nil
}
func (p Parameters) GenEvaluationKeys(sk *rlwe.SecretKey) (*Keys, *rlwe.SecretKey, error) {
	smallSk := rlwe.NewKeyGenerator(p.small).GenSecretKeyNew()
	gen := rlwe.NewKeyGenerator(p.source)
	refresh, _, err := p.circuit.GenEvaluationKeys(smallSk)
	if err != nil {
		return nil, nil, err
	}
	return &Keys{gen.GenEvaluationKeyNew(sk, smallSk), gen.GenEvaluationKeyNew(smallSk, sk), refresh}, nil, nil
}
func NewEvaluator(p Parameters, k *Keys) (*Evaluator, error) {
	down, err := backend.NewSparseDownConverter(p.source, p.small, k.down)
	if err != nil {
		return nil, err
	}
	refresh, err := boot.NewEvaluator(p.circuit, k.refresh)
	if err != nil {
		return nil, err
	}
	return &Evaluator{p: p, down: down, up: rlwe.NewEvaluator(p.source, nil), upKey: k.up, refresh: refresh}, nil
}
func (e *Evaluator) Bootstrap(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	p := e.p.plan
	if ct == nil || ct.MetaData == nil || ct.Level() != p.Level || ct.Value[0].N() != p.Source || !ct.IsBatched || ct.LogDimensions.Rows != 0 ||
		ct.LogDimensions.Cols != bits.TrailingZeros(uint(p.Period)) || ct.Degree() != 1 || !(math.Abs(ct.LogScale()-float64(p.LogScale)) <= 1e-6) {
		return nil, fmt.Errorf("ring-bootstrap input level/scale/layout mismatch")
	}
	var obs Observation
	note := func(i int, x *rlwe.Ciphertext) {
		obs.Levels[i] = x.Level()
		obs.Rings[i] = x.Value[0].N()
		obs.LogScales[i] = x.LogScale()
	}
	note(0, ct)
	start := time.Now()
	aligned := ct.CopyNew()
	aligned.Resize(1, p.Level-1)
	small, err := e.down.Convert(aligned)
	if err != nil {
		return nil, err
	}
	obs.SegmentNS[0] = time.Since(start).Nanoseconds()
	note(1, small)
	start = time.Now()
	fresh, err := e.refresh.Bootstrap(small)
	if err != nil {
		return nil, err
	}
	obs.SegmentNS[1] = time.Since(start).Nanoseconds()
	note(2, fresh)
	if fresh.Value[0].N() != p.Target || fresh.Level() != e.p.small.MaxLevel() || fresh.LogDimensions != ct.LogDimensions {
		return nil, fmt.Errorf("unexpected small-ring refresh output")
	}
	start = time.Now()
	out := ckks.NewCiphertext(e.p.source, 1, fresh.Level())
	if err = e.up.ApplyEvaluationKey(fresh, e.upKey, out); err != nil {
		return nil, err
	}
	obs.SegmentNS[2] = time.Since(start).Nanoseconds()
	note(3, out)
	e.Last = obs
	return out, nil
}
