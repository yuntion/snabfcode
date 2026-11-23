package epochs

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestDiagonalRootGuard(t *testing.T) {
	const m, p = 32768, 64
	r := ckks.GetRootsComplex128(m)
	if !checkDiagonalRoots(r, m, p) || !checkDiagonalRoots(r, m, 4) {
		t.Fatal("rejected real roots")
	}
	if checkDiagonalRoots(r[:m], m, p) || checkDiagonalRoots(r, m, 3) {
		t.Fatal("accepted unbound geometry")
	}
	for _, tc := range []struct {
		i int
		z complex128
	}{{m / (4 * p), r[m/(4*p)] + 0x1p-44i}, {1, complex(math.NaN(), 0)}, {m, -1}} {
		save := r[tc.i]
		r[tc.i] = tc.z
		if checkDiagonalRoots(r, m, p) {
			t.Fatal("accepted stale root", tc.i)
		}
		r[tc.i] = save
	}
}
