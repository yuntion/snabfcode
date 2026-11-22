package epochs

import (
	"math"
	"math/big"
)

// A distinct generated symbol: linking new protected code against the old
// backend fails. Old Bind/RequireExactEncoding binaries acquire no such proof.
const RequireDiagonalEncodingV1 EncodingPolicy = 2

// HasDiagonalRootGuard distinguishes a successful binding from an old
// executable with identical plan and public-value digests.
func (r *Runtime) HasDiagonalRootGuard(n int) bool { return r.diagonalRoots[n] }

// Check the actual DIF twiddles against mathematical roots without trusting a
// libm accuracy promise. Off-grid roots occur only in zero differences during
// exact tile folding; even those must be finite. This is a guard on every
// bound runtime, not a calibration run or a probabilistic premise.
func checkDiagonalRoots(roots []complex128, m, p int) bool {
	if p < 2 || p > 64 || p&(p-1) != 0 || m < 4*p || m%(4*p) != 0 || len(roots) != m+1 {
		return false
	}
	for _, z := range roots {
		if !finite(real(z)) || !finite(imag(z)) {
			return false
		}
	}
	// c[k] = cos(k*pi/(2p)). Enclose in units 2^-96 using the positive
	// half-angle sqrt, reflecting cos(2 theta) through pi/2 as necessary.
	type interval struct{ l, h *big.Int }
	one := big.NewInt(1)
	f := new(big.Int).Lsh(one, 96)
	cs := make([]interval, p+1)
	cs[0], cs[p] = interval{f, f}, interval{big.NewInt(0), big.NewInt(0)}
	half := func(x *big.Int) *big.Int {
		t := new(big.Int).Add(f, x)
		return t.Sqrt(t.Lsh(t, 95))
	}
	for step := p / 2; step > 0; step /= 2 {
		for k := step; k < p; k += 2 * step {
			j := 2 * k
			var c interval
			if j > p {
				a := cs[2*p-j]
				c = interval{new(big.Int).Neg(a.h), new(big.Int).Neg(a.l)}
			} else {
				c = cs[j]
			}
			cs[k] = interval{half(c.l), new(big.Int).Add(half(c.h), one)}
		}
	}
	component := func(k int, x float64) bool {
		k %= 4 * p
		if k > 2*p {
			k = 4*p - k
		}
		if k > p {
			k, x = 2*p-k, -x
		}
		c := cs[k]
		// Outward integer endpoints in units 2^-50. Accept the INNER
		// interval [hi-4,lo+4]; every possible true value is within 2^-48.
		lo := new(big.Int).Rsh(c.l, 46).Int64()
		hi := new(big.Int).Rsh(c.h, 46).Int64() + 1
		return x >= math.Ldexp(float64(hi-4), -50) && x <= math.Ldexp(float64(lo+4), -50)
	}
	for k := 0; k < 4*p; k++ {
		z := roots[k*(m/(4*p))]
		if !component(k, real(z)) || !component(k+3*p, imag(z)) {
			return false
		}
	}
	// Lattigo stores the periodic endpoint as a separate table entry.
	return roots[m] == roots[0]
}
