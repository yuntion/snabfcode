package tower_test

import (
	"math"
	"testing"

	"snabf.local/compiler/src/backend/tower"
)

func TestFullSlotBlockCoordinates(t *testing.T) {
	const half, rows, cols = 4, 6, 7
	d := make(tower.Diagonals[float64])
	for k := range 2 * half {
		d[k] = make([]float64, 2*half)
		for i := range rows {
			j := (i + k) % (2 * half)
			if j < cols {
				d[k][i] = float64(100*i + j + 1)
			}
		}
	}
	b, err := tower.PartitionFullSlotDiagonals(d, half, cols, rows)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 2 * half {
		for j := range 2 * half {
			var want, got float64
			if i < rows && j < cols {
				want = float64(100*i + j + 1)
			}
			if band := b[i/half][j/half][(j-i+2*half)%half]; band != nil {
				got = band[i%half]
			}
			if got != want {
				t.Fatalf("W[%d,%d]=%g, want %g", i, j, got, want)
			}
		}
	}
	// Sparse matrices do not manufacture dummy ciphertext products or rotation keys.
	sparse, err := tower.PartitionFullSlotDiagonals(tower.Diagonals[float64]{0: {1, 0, 0, 0, 0, 0, 0, 0}}, half, cols, rows)
	if err != nil || len(sparse[0][0]) != 1 || len(sparse[0][1])+len(sparse[1][0])+len(sparse[1][1]) != 0 {
		t.Fatal(sparse, err)
	}
	for _, bad := range []tower.Diagonals[float64]{
		{0: {0}}, {-1: make([]float64, 8)}, {8: make([]float64, 8)}, {0: {math.NaN(), 0, 0, 0, 0, 0, 0, 0}},
		{0: {1, 0, 0, 0, 0, 0, 1, 0}}, {7: {1, 0, 0, 0, 0, 0, 0, 0}},
	} {
		if _, err := tower.PartitionFullSlotDiagonals(bad, half, cols, rows); err == nil {
			t.Fatalf("admitted invalid full diagonal: %v", bad)
		}
	}
}
