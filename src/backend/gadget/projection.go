package gadget

import (
	"math"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/ring"
)

// shortCRT uses mixed-radix digits in the ORIGINAL order of the prefix primes.
// Constants are local to the actual moduli/level of this projection. No big.Int
// or heap allocation is needed in its coefficient loop, even for 240-bit D.
type shortCRT struct {
	q       []*ring.SubRing
	inv     [4][4]uint64
	targets []shortTarget
}

type shortTarget struct {
	sr    *ring.SubRing
	radix [3]uint64 // q[i] in Montgomery form modulo this retained prime
	d     uint64    // D modulo this prime (ordinary representation)
}

func newShortCRT(rq *ring.Ring, k int) shortCRT {
	p := shortCRT{q: rq.SubRings[:k]}
	for i, sr := range p.q {
		q := sr.Modulus
		for j := 0; j < i; j++ {
			inv := ring.ModExp(p.q[j].Modulus%q, q-2, q)
			p.inv[i][j] = ring.MForm(inv, q, sr.BRedConstant)
		}
	}
	for _, sr := range rq.SubRings[k : rq.Level()+1] {
		t := shortTarget{sr: sr, d: 1}
		for i, src := range p.q {
			w := ring.MForm(src.Modulus%sr.Modulus, sr.Modulus, sr.BRedConstant)
			if i+1 < k {
				t.radix[i] = w
			}
			t.d = ring.MRed(t.d, w, sr.Modulus, sr.MRedConstant)
		}
		p.targets = append(p.targets, t)
	}
	return p
}

func (p *shortCRT) digits(a ring.Poly, j int) (d [4]uint64, neg bool) {
	for i, sr := range p.q {
		q := sr.Modulus
		x := a.Coeffs[i][j]
		for l := 0; l < i; l++ {
			dl := ring.BRedAdd(d[l], q, sr.BRedConstant)
			x = ring.MRed(subMod(x, dl, q), p.inv[i][l], q, sr.MRedConstant)
		}
		d[i] = x
		// For odd radices the mixed-radix digits of floor(D/2) are q[i]/2.
		// The most significant unequal digit wins; exact equality is positive.
		if x != q/2 {
			neg = x > q/2
		}
	}
	return
}

func subMod(x, y, q uint64) uint64 { return ring.CRed(x+q-y, q) }

func (p *shortCRT) apply(a ring.Poly, stats *moments) {
	k := len(p.q)
	for j := range a.Coeffs[0] {
		d, neg := p.digits(a, j)
		for l, t := range p.targets {
			s, q := t.sr, t.sr.Modulus
			r := ring.BRedAdd(d[k-1], q, s.BRedConstant)
			for i := k - 2; i >= 0; i-- {
				r = ring.CRed(ring.MRed(r, t.radix[i], q, s.MRedConstant)+ring.BRedAdd(d[i], q, s.BRedConstant), q)
			}
			if neg {
				r = subMod(r, t.d, q)
			}
			a.Coeffs[k+l][j] = subMod(a.Coeffs[k+l][j], r, q)
		}
		if stats != nil {
			var f float64
			for i := k - 1; i >= 0; i-- {
				x := d[i]
				if neg {
					x = p.q[i].Modulus - 1 - x // digits of D-1-x; preserves tiny negative residues
				}
				f = f*float64(p.q[i].Modulus) + float64(x)
			}
			if neg {
				f = -1 - f
			}
			stats.add(f)
		}
	}
}

func projectBig(rq *ring.Ring, a ring.Poly, k int, stats *moments) {
	d := rq.ModulusAtLevel[k-1]
	half := new(big.Int).Rsh(new(big.Int).Set(d), 1)
	mods := make([]big.Int, rq.Level()+1)
	basis := make([]big.Int, k)
	var tmp, z big.Int
	for i := range mods {
		mods[i].SetUint64(rq.SubRings[i].Modulus)
		if i < k {
			basis[i].Quo(d, &mods[i])
			tmp.ModInverse(&basis[i], &mods[i])
			basis[i].Mul(&basis[i], &tmp)
		}
	}
	for j := range a.Coeffs[0] {
		z.SetUint64(0)
		for i := range basis {
			tmp.SetUint64(a.Coeffs[i][j])
			z.Add(&z, tmp.Mul(&tmp, &basis[i]))
		}
		z.Mod(&z, d)
		if z.Cmp(half) > 0 {
			z.Sub(&z, d)
		}
		for i := k; i < len(mods); i++ {
			r := tmp.Mod(&z, &mods[i]).Uint64()
			a.Coeffs[i][j] = subMod(a.Coeffs[i][j], r, mods[i].Uint64())
		}
		if stats != nil {
			f, _ := z.Float64()
			stats.add(f)
		}
	}
}

type moments struct{ peak, sum, squares float64 }

func (m *moments) add(x float64) {
	m.peak = math.Max(m.peak, math.Abs(x))
	m.sum += x
	m.squares += x * x
}

// project returns coefficient-domain x-center_D(x); no quotient, index shift,
// level drop or scale change. The ORIGINAL prefix needs its INTT to recover r.
func project(rq *ring.Ring, in ring.Poly, k int, ev *Event) ring.Poly {
	a := rq.NewPoly()
	for i, sr := range rq.SubRings[:rq.Level()+1] {
		sr.INTT(in.Coeffs[i], a.Coeffs[i])
		ev.InverseNTT++
	}
	var m *moments
	if Observe != nil {
		m = new(moments)
	}
	if k <= 4 {
		p := newShortCRT(rq, k)
		p.apply(a, m)
	} else {
		projectBig(rq, a, k, m)
	}
	for _, limb := range a.Coeffs[:k] {
		clear(limb)
		ev.TruncatedLimbsK++
	}
	if m != nil {
		n := float64(rq.N())
		ev.ResidualMax, ev.ResidualMean, ev.ResidualMeanSquare = m.peak, m.sum/n, m.squares/n
		ev.ResidualVariance = math.Max(0, ev.ResidualMeanSquare-ev.ResidualMean*ev.ResidualMean)
	}
	return a
}
