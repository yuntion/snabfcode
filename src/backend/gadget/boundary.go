package gadget

import (
	"math/big"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
)

// A partially zero digit still has the modulus M of the ENTIRE original block.
// v6.2.0 extends a+floor(M/2) by CRT, then subtracts floor(M/2). For its
// leading zero inputs the shifted CRT terms are constants, not zero. Fold them
// (and their contribution to the native floating quotient) once, in limb order.
// Only the live inputs enter the coefficient loop. The result agrees modulo
// each destination with DecomposeSingleNTT, including its rounding convention.
func extendBoundary(qp ringqp.Ring, a ring.Poly, lo, k, end int, dst ringqp.Poly) {
	q := qp.RingQ.SubRings[lo:end]
	var m, half, tmp, modulus big.Int
	m.SetUint64(1)
	for _, s := range q {
		m.Mul(&m, tmp.SetUint64(s.Modulus))
	}
	half.Rsh(&m, 1)
	var stars [4]big.Int
	var inv, frozen [4]uint64
	var prefixQuotient float64
	for i, s := range q {
		modulus.SetUint64(s.Modulus)
		stars[i].Quo(&m, &modulus)
		inv[i] = ring.MForm(tmp.ModInverse(&stars[i], &modulus).Uint64(), s.Modulus, s.BRedConstant)
		if lo+i < k {
			frozen[i] = ring.MRed(s.Modulus/2, inv[i], s.Modulus, s.MRedConstant)
			prefixQuotient += float64(frozen[i]) / float64(s.Modulus)
		}
	}
	type target struct {
		s       *ring.SubRing
		out     []uint64
		weights [4]uint64
		bias    [5]uint64 // indexed by the native CRT correction v, including v == len(q)
	}
	var targets []target
	addTarget := func(s *ring.SubRing, out []uint64) {
		t := target{s: s, out: out}
		mod := s.Modulus
		modulus.SetUint64(mod)
		mMod := tmp.Mod(&m, &modulus).Uint64()
		bias := subMod(0, tmp.Mod(&half, &modulus).Uint64(), mod)
		for i := range q {
			w := ring.MForm(tmp.Mod(&stars[i], &modulus).Uint64(), mod, s.BRedConstant)
			if lo+i < k {
				bias = ring.CRed(bias+ring.MRed(frozen[i], w, mod, s.MRedConstant), mod)
			} else {
				t.weights[i] = w
			}
		}
		for i := 0; i <= len(q); i++ {
			t.bias[i] = bias
			bias = subMod(bias, mMod, mod)
		}
		targets = append(targets, t)
	}
	for i, s := range qp.RingQ.SubRings[:qp.LevelQ()+1] {
		if i < lo || i >= end {
			addTarget(s, dst.Q.Coeffs[i])
		}
	}
	for i, s := range qp.RingP.SubRings[:qp.LevelP()+1] {
		addTarget(s, dst.P.Coeffs[i])
	}
	for j := range a.Coeffs[k] {
		var y [4]uint64
		v := prefixQuotient
		for i := k - lo; i < len(q); i++ {
			s := q[i]
			y[i] = ring.MRed(a.Coeffs[lo+i][j]+s.Modulus/2, inv[i], s.Modulus, s.MRedConstant)
			v += float64(y[i]) / float64(s.Modulus)
		}
		for i := range targets {
			t := &targets[i]
			z := t.bias[int(v)]
			for l := k - lo; l < len(q); l++ {
				z = ring.CRed(z+ring.MRed(y[l], t.weights[l], t.s.Modulus, t.s.MRedConstant), t.s.Modulus)
			}
			t.out[j] = z
		}
	}
}
