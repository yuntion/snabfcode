package gadget

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
)

type transformCounts struct{ forward, inverse int }

type countedNTT struct {
	s *ring.SubRing
	c *transformCounts
}

func (n countedNTT) Forward(a, b []uint64)      { n.s.NTT(a, b); n.c.forward++ }
func (n countedNTT) ForwardLazy(a, b []uint64)  { n.s.NTTLazy(a, b); n.c.forward++ }
func (n countedNTT) Backward(a, b []uint64)     { n.s.INTT(a, b); n.c.inverse++ }
func (n countedNTT) BackwardLazy(a, b []uint64) { n.s.INTTLazy(a, b); n.c.inverse++ }

// Only for privately owned diagnostic parameters; never wrap a caller's ring.
// The delegate retains the concrete transform and precomputed table (including
// ring type). All four Lattigo subring NTT entry points execute normally.
func countTransforms(r *ring.Ring, c *transformCounts) error {
	for i, s := range r.SubRings {
		wrapped, err := ring.NewSubRingWithCustomNTT(s.N, s.Modulus, func(*ring.SubRing, int) ring.NumberTheoreticTransformer {
			return countedNTT{s, c}
		}, int(s.NthRoot))
		if err != nil {
			return err
		}
		wrapped.NTTTable, wrapped.Factors = s.NTTTable, s.Factors
		r.SubRings[i] = wrapped
	}
	return nil
}

// In an observed run only, execute the native baseline on the SAME input/key
// under a private evaluator. Stop before the common ModDown. A hoisted baby
// group pays DecomposeNTT once; a direct rotation/relin pays GadgetProductLazy
// (not the same transformer call pattern for a one-P key). Never infer native
// work from alpha, dimensions, skipped blocks or a cost-model coefficient.
func measureNative(p *rlwe.Parameters, l int, in ring.Poly, key *rlwe.GadgetCiphertext, shared bool) (c transformCounts, err error) {
	private, err := rlwe.NewParametersFromLiteral(p.ParametersLiteral())
	if err != nil {
		return c, err
	}
	e := rlwe.NewEvaluator(private, nil)
	for _, r := range []*ring.Ring{private.RingQ(), private.RingP()} {
		if err = countTransforms(r, &c); err != nil {
			return c, err
		}
	}
	lp := key.LevelP()
	if shared {
		qp := private.RingQP().AtLevel(l, lp)
		rows := make([]ringqp.Poly, private.BaseRNSDecompositionVectorSize(l, lp))
		for i := range rows {
			rows[i] = qp.NewPoly()
		}
		e.DecomposeNTT(l, lp, lp+1, in, true, rows)
	} else {
		acc := rlwe.NewElementExtended(private, 1, l, lp)
		acc.IsNTT = true
		err = e.GadgetProductLazy(l, in, key, acc)
	}
	return c, err
}
