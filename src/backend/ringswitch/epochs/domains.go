package epochs

import (
	"fmt"
	"math/bits"
	"slices"
)

type Layout struct {
	N, Capacity, Width, Period, Stride, Level, Lanes int
	Packing                                          string
	LogScale                                         float64
	Offsets, Active, Mask                            []int
}

type Transition struct {
	FromKey, ToKey                     string
	LevelQ, LevelP, KeyBytes, Parallel int
	Operands                           []CutOperand
	Basis                              SwitchBasis
}

type CutOperand struct {
	SSA, Edge, KeyID, User string
	UserIndex, Operand     int
	Provided, Required     Layout // physical layouts at the call, after any use-local drop
	Consumers              []CutUse
}
type CutUse struct {
	User               string
	UserIndex, Operand int
}
type SwitchBasis struct {
	Ring, EmbeddedRing int
	Q, P               []uint64
}

func (x Layout) periodic(n int, s Program) bool {
	return x.N == n && x.Capacity == n/2 && s.Slots <= x.Capacity && x.Width == s.Width &&
		x.Period == s.Slots && x.Stride == 1 && x.Lanes == 1 && x.Packing == "periodic_single" &&
		slices.Equal(x.Offsets, []int{0}) && slices.Equal(x.Active, []int{0}) && len(x.Mask) == 0 &&
		finite(x.LogScale) && x.LogScale > 0
}

func validateDomains(s Program) error {
	domains := map[int]Epoch{}
	ids := map[string]bool{}
	for _, e := range s.Epochs {
		if e.N < 4 || bits.OnesCount(uint(e.N)) != 1 || (e.N != s.Source && e.N != s.Target) || domains[e.N].N != 0 || e.KeyID == "" || ids[e.KeyID] || len(e.Q) == 0 || len(e.P) == 0 || e.Scale <= 0 || s.Slots > e.N/2 {
			return fmt.Errorf("invalid key domain N=%d", e.N)
		}
		seen := map[uint64]bool{}
		for _, g := range e.Galois {
			if g <= 1 || g%2 == 0 || g >= 2*uint64(e.N) || seen[g] {
				return fmt.Errorf("invalid Galois key %d in %s", g, e.KeyID)
			}
			seen[g] = true
		}
		domains[e.N], ids[e.KeyID] = e, true
	}
	a, b := domains[s.Source], domains[s.Target]
	if a.N == 0 || b.N == 0 || (s.Switch != nil) != (a.N != b.N) {
		return fmt.Errorf("missing or surplus conversion binding")
	}
	if t := s.Switch; t != nil {
		if len(t.Operands) < 1 || len(t.Operands) > 2 || t.Basis.Ring != a.N || t.Basis.EmbeddedRing != b.N || !slices.Equal(t.Basis.Q, b.Q) || !slices.Equal(t.Basis.P, b.P) {
			return fmt.Errorf("conversion cut/extended-key basis mismatch")
		}
		if a.N <= b.N || t.Parallel != 1 || t.FromKey != a.KeyID || t.ToKey != b.KeyID || t.LevelQ != len(b.Q)-1 || t.LevelP != len(b.P)-1 || t.KeyBytes <= 0 || len(a.Q) < len(b.Q) || !slices.Equal(a.Q[:len(b.Q)], b.Q) || !slices.Equal(a.P, b.P) {
			return fmt.Errorf("conversion layout/key/basis mismatch")
		}
		ssa, edges, users := map[string]bool{}, map[string]bool{}, map[CutUse]bool{}
		for i, op := range t.Operands {
			if op.SSA == "" || ssa[op.SSA] || op.Edge == "" || edges[op.Edge] || op.KeyID != a.KeyID || op.User != "snabf.convert" || op.Operand != 0 || op.UserIndex < 0 || op.UserIndex != t.Operands[0].UserIndex+i ||
				!op.Provided.periodic(a.N, s) || !op.Required.periodic(b.N, s) || op.Provided.Level != t.LevelQ || op.Required.Level != t.LevelQ || op.Provided.LogScale != op.Required.LogScale || (len(t.Operands) > 1 && len(op.Consumers) == 0) {
				return fmt.Errorf("invalid conversion operand %d", i)
			}
			ssa[op.SSA], edges[op.Edge] = true, true
			for _, u := range op.Consumers {
				if users[u] || u.UserIndex <= op.UserIndex || !((u.User == "snabf.lintrans" && u.Operand == 0) || (u.User == "snabf.ckks_step" && (u.Operand == 0 || u.Operand == 1))) {
					return fmt.Errorf("invalid conversion use")
				}
				users[u] = true
			}
		}
	}
	return nil
}
