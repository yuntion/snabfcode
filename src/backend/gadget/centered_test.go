package gadget

import (
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math/big"
	"testing"
)

func TestPrefixProjection(t *testing.T) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 10, LogQ: []int{21, 24, 25}, LogP: []int{35}, LogDefaultScale: 24})
	if err != nil {
		t.Fatal(err)
	}
	rq := p.RingQ()
	qs := p.Q()
	for _, k := range []int{1, 2} {
		D := big.NewInt(1)
		for _, q := range qs[:k] {
			D.Mul(D, new(big.Int).SetUint64(q))
		}
		half := new(big.Int).Rsh(new(big.Int).Set(D), 1)
		a := rq.NewPoly()
		zs := []*big.Int{big.NewInt(0), half, new(big.Int).Add(half, big.NewInt(1)), new(big.Int).Neg(half), new(big.Int).Add(D, big.NewInt(7))}
		for j, z := range zs {
			for i, q := range qs {
				a.Coeffs[i][j] = new(big.Int).Mod(z, new(big.Int).SetUint64(q)).Uint64()
			}
		}
		rq.NTT(a, a)
		ev := Event{}
		r := project(rq, a, k, &ev)
		for j, z := range zs {
			center := new(big.Int).Mod(z, D)
			if center.Cmp(half) > 0 {
				center.Sub(center, D)
			}
			want := new(big.Int).Sub(z, center)
			for i, q := range qs {
				w := new(big.Int).Mod(want, new(big.Int).SetUint64(q)).Uint64()
				if r.Coeffs[i][j] != w {
					t.Fatalf("k=%d limb=%d z=%v got=%d want=%d", k, i, z, r.Coeffs[i][j], w)
				}
			}
		}
	}
}
