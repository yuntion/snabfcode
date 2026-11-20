package gadget

import (
	"math/big"
	"math/rand"
	"testing"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestPrefixProjection(t *testing.T) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 8, LogQ: []int{59, 57, 58, 56, 29, 50, 24}, LogP: []int{55, 55}, LogDefaultScale: 40})
	if err != nil {
		t.Fatal(err)
	}
	rq := p.RingQ()
	qs := p.Q()
	rng := rand.New(rand.NewSource(713))
	for k := 1; k <= rq.Level(); k++ {
		D := big.NewInt(1)
		for _, q := range qs[:k] {
			D.Mul(D, new(big.Int).SetUint64(q))
		}
		half := new(big.Int).Rsh(new(big.Int).Set(D), 1)
		a := rq.NewPoly()
		zs := []*big.Int{big.NewInt(0), half, new(big.Int).Add(half, big.NewInt(1)), new(big.Int).Neg(half), new(big.Int).Add(D, big.NewInt(7)), big.NewInt(-1), new(big.Int).Sub(D, big.NewInt(1))}
		for len(zs) < rq.N() {
			zs = append(zs, new(big.Int).Rand(rng, rq.ModulusAtLevel[rq.Level()]))
		}
		for j, z := range zs {
			for i, q := range qs {
				a.Coeffs[i][j] = new(big.Int).Mod(z, new(big.Int).SetUint64(q)).Uint64()
			}
		}
		rq.NTT(a, a)
		ev := Event{}
		r := project(rq, a, k, &ev)
		if ev.TruncatedLimbsK != k || ev.InverseNTT != p.QCount() {
			t.Fatalf("k=%d unexpected projection counts: %+v", k, ev)
		}
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

// Independent dense oracle: lift the WHOLE polynomial to Z_Q, project the
// integers directly, then send the result back through the ordinary native API.
func oracleNTT(rq *ring.Ring, in ring.Poly, k int) ring.Poly {
	a := rq.NewPoly()
	rq.INTT(in, a)
	z := make([]*big.Int, rq.N())
	rq.PolyToBigint(a, 1, z)
	d := rq.ModulusAtLevel[k-1]
	half := new(big.Int).Rsh(new(big.Int).Set(d), 1)
	var r big.Int
	for _, x := range z {
		r.Mod(x, d)
		if r.Cmp(half) > 0 {
			r.Sub(&r, d)
		}
		x.Sub(x, &r)
	}
	rq.SetCoefficientsBigint(z, a)
	rq.NTT(a, a)
	return a
}

func TestShortCRTNoCoefficientAllocations(t *testing.T) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 8, LogQ: []int{57, 59, 55, 58, 27}, LogP: []int{29, 30}, LogDefaultScale: 25})
	if err != nil {
		t.Fatal(err)
	}
	a := p.RingQ().NewPoly()
	for i, s := range p.RingQ().SubRings {
		for j := range a.Coeffs[i] {
			a.Coeffs[i][j] = s.Modulus - 1 - uint64(j)
		}
	}
	crt := newShortCRT(p.RingQ(), 4)
	if n := testing.AllocsPerRun(10, func() { crt.apply(a, nil) }); n != 0 {
		t.Fatalf("four-limb Garner coefficient loop allocated %g times", n)
	}
}
