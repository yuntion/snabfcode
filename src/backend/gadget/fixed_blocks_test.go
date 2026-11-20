package gadget

import (
	"math/big"
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
