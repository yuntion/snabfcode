package gadget

import (
	"fmt"
	"math/big"
	"math/rand"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Compare actual Lattigo decomposition digits after projection, not two
// implementations of ceil(). D+1 projects to D, nonzero in EVERY retained limb.
func TestTwoPFixedKeyBlocks(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 10, LogQ: []int{30, 31, 32, 33, 34}, LogP: []int{35, 36}, LogDefaultScale: 30})
	if err != nil {
		t.Fatal(err)
	}
	ev := ckks.NewEvaluator(params, nil)
	for level := 0; level < params.QCount(); level++ {
		rq := params.RingQ().AtLevel(level)
		rqp := params.RingQP().AtLevel(level, 1)
		key := rlwe.NewEvaluationKey(params, rlwe.EvaluationKeyParameters{LevelQ: &level})
		for k := 0; k <= level; k++ {
			d := big.NewInt(1)
			for _, q := range params.Q()[:k] {
				d.Mul(d, new(big.Int).SetUint64(q))
			}
			x := new(big.Int).Add(d, big.NewInt(1))
			a := rq.NewPoly()
			for i, q := range params.Q()[:level+1] {
				a.Coeffs[i][0] = new(big.Int).Mod(x, new(big.Int).SetUint64(q)).Uint64()
			}
			if k != 0 {
				rq.NTT(a, a)
				a = project(rq, a, k, &Event{})
			}
			for j := 0; j <= level; j++ {
				if (a.Coeffs[j][0] == 0) != (j < k) {
					t.Fatalf("L%d k=%d: projected limb %d", level, k, j)
				}
			}
			ntt := rq.NewPoly()
			rq.NTT(a, ntt)
			zero := rqp.NewPoly()
			count := 0
			for b := range key.Value {
				row := rqp.NewPoly()
				// DecomposeSingleNTT also restores this digit's native Q limbs.
				ev.DecomposeSingleNTT(level, 1, 2, b, ntt, a, row.Q, row.P)
				// Lazy modular zeros need not be represented by literal 0.
				rqp.Reduce(row, row)
				active := !rqp.Equal(row, zero)
				if active {
					count++
				}
				if active != (b >= k/2) {
					t.Fatalf("L%d k=%d: physical key block %d active=%t", level, k, b, active)
				}
			}
			if count != 1+level/2-k/2 {
				t.Fatalf("L%d k=%d: %d active blocks", level, k, count)
			}
		}
	}
}

// Check every Q/P limb, not just a nonzero marker. At lower levels, the
// evaluator still uses the full-level key partition, including a short tail.
func TestAllFixedDigitsAgainstLattigo(t *testing.T) {
	for alpha := 1; alpha <= 4; alpha++ {
		t.Run(fmt.Sprintf("alpha%d", alpha), func(t *testing.T) {
			p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 8, LogQ: []int{59, 27, 55, 31, 56, 58, 25}, LogP: []int{53, 54, 54, 55}[:alpha], LogDefaultScale: 30})
			if err != nil {
				t.Fatal(err)
			}
			e := ckks.NewEvaluator(p, nil)
			rng := rand.New(rand.NewSource(926))
			for l := 1; l <= p.MaxLevel(); l++ {
				qp := p.RingQP().AtLevel(l, alpha-1)
				rq := qp.RingQ
				in := rq.NewPoly()
				for i, s := range rq.SubRings[:l+1] {
					for j := range in.Coeffs[i] {
						in.Coeffs[i][j] = rng.Uint64() % s.Modulus
					}
				}
				rq.NTT(in, in)
				for k := 1; k <= l; k++ {
					h, err := decompose(e, in, Site{Level: l, K: k})
					if err != nil {
						t.Fatal(err)
					}
					ntt := oracleNTT(rq, in, k)
					a := rq.NewPoly()
					rq.INTT(ntt, a)
					if len(h.rows) != (l+alpha)/alpha {
						t.Fatal("physical key rows were reindexed")
					}
					for b, got := range h.rows {
						want := qp.NewPoly()
						e.DecomposeSingleNTT(l, alpha-1, alpha, b, ntt, a, want.Q, want.P)
						qp.Reduce(want, want)
						if got.Q.Coeffs == nil {
							if b >= k/alpha || !qp.Equal(want, qp.NewPoly()) {
								t.Fatalf("L%d k%d: skipped a live physical block %d", l, k, b)
							}
						} else if b < k/alpha || !qp.Equal(got, want) {
							t.Fatalf("L%d k%d: Q/P digit mismatch in physical block %d", l, k, b)
						}
					}
				}
			}
		})
	}
}

// With Z the frozen part of the digit and M=Z*R its full modulus, the two
// closest representable projected values on either side of M/2 are
// Z*floor(R/2) and Z*(floor(R/2)+1). With 60-bit primes their normalized
// distance can be smaller than a float64 ulp in Lattigo's quotient correction.
func TestPartialDigitNearNativeHalf(t *testing.T) {
	for _, test := range []struct{ alpha, level, block int }{{2, 4, 1}, {3, 5, 0}, {3, 4, 1}, {4, 6, 1}, {4, 7, 0}} {
		alpha, level, b := test.alpha, test.level, test.block
		t.Run(fmt.Sprintf("a%d/L%d/b%d", alpha, level, b), func(t *testing.T) {
			p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 8, LogQ: []int{60, 59, 58, 60, 59, 58, 60, 59}, LogP: []int{60, 59, 58, 57}[:alpha], LogDefaultScale: 40})
			if err != nil {
				t.Fatal(err)
			}
			e := ckks.NewEvaluator(p, nil)
			qp := p.RingQP().AtLevel(level, alpha-1)
			lo, end := b*alpha, min((b+1)*alpha, level+1)
			for k := lo + 1; k < end; k++ {
				z, r := big.NewInt(1), big.NewInt(1)
				for i := lo; i < end; i++ {
					prod := r
					if i < k {
						prod = z
					}
					prod.Mul(prod, new(big.Int).SetUint64(p.Q()[i]))
				}
				r.Rsh(r, 1)
				a := qp.RingQ.NewPoly()
				for j := 0; j < p.N(); j++ {
					x := new(big.Int).Add(r, big.NewInt(int64(j%6-2)))
					x.Mul(x, z)
					// All prefix residues are already zero, so projection is the identity here.
					for i := k; i <= level; i++ {
						a.Coeffs[i][j] = new(big.Int).Mod(x, new(big.Int).SetUint64(p.Q()[i])).Uint64()
					}
				}
				ntt := qp.RingQ.NewPoly()
				qp.RingQ.NTT(a, ntt)
				h, err := decompose(e, ntt, Site{Level: level, K: k})
				if err != nil {
					t.Fatal(err)
				}
				got, want := h.rows[b], qp.NewPoly()
				if got.Q.Coeffs == nil || h.info.PartialBlockZeroLimbs != k-lo {
					t.Fatalf("k=%d: lost physical boundary digit", k)
				}
				e.DecomposeSingleNTT(level, alpha-1, alpha, b, ntt, a, want.Q, want.P)
				qp.Reduce(got, got)
				qp.Reduce(want, want)
				if !qp.Equal(got, want) {
					t.Fatalf("k=%d: optimized Q/P digit differs at the original block half", k)
				}
			}
		})
	}
}
