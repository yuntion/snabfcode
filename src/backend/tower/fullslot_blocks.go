package tower

import (
	"fmt"
	"math"
)

// PartitionFullSlotDiagonals partitions one standard full-capacity rotation
// layout into [output half][input half]. Diagonal k stores W[i,(i+k)%slots]
// at i. The source is never modified; empty child maps mean exact zero blocks.
// This transforms public weights only; it is not a cut or key-domain receipt.
func PartitionFullSlotDiagonals(src Diagonals[float64], childSlots, inputWidth, outputWidth int) ([2][2]Diagonals[float64], error) {
	var blocks [2][2]Diagonals[float64]
	if childSlots < 1 || childSlots&(childSlots-1) != 0 || childSlots > math.MaxInt/2 {
		return blocks, fmt.Errorf("expected a power-of-two target slot count")
	}
	full := 2 * childSlots
	if inputWidth <= childSlots || inputWidth > full || outputWidth <= 0 || outputWidth > full {
		return blocks, fmt.Errorf("invalid logical extent for a 1-to-2 full-slot split")
	}
	for k, diag := range src {
		if k < 0 || k >= full || len(diag) != full {
			return [2][2]Diagonals[float64]{}, fmt.Errorf("invalid full diagonal %d or mask size", k)
		}
		for row, w := range diag {
			if w == 0 {
				continue
			}
			col := (row + k) & (full - 1)
			if row >= outputWidth || col >= inputWidth || math.IsNaN(w) || math.IsInf(w, 0) {
				return [2][2]Diagonals[float64]{}, fmt.Errorf("invalid coefficient at output %d input %d", row, col)
			}
			block := &blocks[row/childSlots][col/childSlots]
			if *block == nil {
				*block = make(Diagonals[float64])
			}
			key := k & (childSlots - 1)
			if (*block)[key] == nil {
				(*block)[key] = make([]float64, childSlots)
			}
			(*block)[key][row&(childSlots-1)] = w
		}
	}
	return blocks, nil
}
