package tower

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Indexed by [output half][input half]. Only a proven zero block has no route.
type FullSlotRoutes [2][2]*Route

// Validate against support derived from the original weights, before calling
// NewTransformation (which assumes its internal caller has checked the route).
func FullSlotRouteKeys(dst ckks.Parameters, level int, block Diagonals[float64], r *Route) ([]uint64, error) {
	if len(block) == 0 && r == nil {
		return nil, nil
	}
	if r == nil || len(block) == 0 || len(r.Terms) != len(block) {
		return nil, fmt.Errorf("missing or surplus child route/diagonal")
	}
	if level < 1 || level > dst.MaxLevel() || r.Level != level || r.LogN != dst.LogN() || r.LogSlots != dst.LogMaxSlots() || r.Babies != nil || r.Giants != nil {
		return nil, fmt.Errorf("unexpected child domain, level or gadget plan")
	}
	s := dst.MaxSlots()
	seen, keys := map[int]bool{}, map[uint64]bool{}
	for _, term := range r.Terms {
		d, b, g := term[0], term[1], term[2]
		if b < 0 || b >= s || g < 0 || g >= s || (b+g)%s != d || seen[d] || len(block[d]) != s {
			return nil, fmt.Errorf("invalid child diagonal or rotation split %v", term)
		}
		seen[d] = true
		for _, rot := range []int{b, g} {
			if rot != 0 {
				keys[dst.GaloisElement(rot)] = true
			}
		}
	}
	return slices.Sorted(maps.Keys(keys)), nil
}

// BindFullSlotJoint consumes snabf-pass1 stdout. Its regions are positional:
// submit exactly the nonzero children in row-major order, all in one scope.
// Neither the declared Galois keys nor the joint byte charge supplies authority.
func BindFullSlotJoint(answer []byte, dst ckks.Parameters, level int, blocks [2][2]Diagonals[float64], scope string, bytesPerKey int) (routes FullSlotRoutes, union []uint64, err error) {
	var a struct {
		Candidates [][]struct {
			Terms  [][]int
			Galois []uint64
		}
		Joint struct {
			Selected []int
			KeyBytes int `json:"key_bytes"`
			Keys     []struct {
				Scope  string
				Galois uint64
				Refs   int
			}
		}
	}
	fail := func(e error) (FullSlotRoutes, []uint64, error) { return FullSlotRoutes{}, nil, e }
	if e := json.Unmarshal(answer, &a); e != nil {
		return fail(e)
	}
	var children [][2]int
	for o := range blocks {
		for i, b := range blocks[o] {
			if len(b) != 0 {
				children = append(children, [2]int{o, i})
			}
		}
	}
	if len(children) == 0 || len(a.Candidates) != len(children) || len(a.Joint.Selected) != len(children) {
		return fail(fmt.Errorf("expected %d selected nonzero child regions", len(children)))
	}
	refs := map[uint64]int{}
	for j, pos := range children {
		s := a.Joint.Selected[j]
		if s < 0 || s >= len(a.Candidates[j]) {
			return fail(fmt.Errorf("missing route for child %v", pos))
		}
		c := a.Candidates[j][s]
		r := &Route{Level: level, LogN: dst.LogN(), LogSlots: dst.LogMaxSlots()}
		for _, term := range c.Terms {
			if len(term) != 3 {
				return fail(fmt.Errorf("malformed term in child %v", pos))
			}
			r.Terms = append(r.Terms, [3]int(term))
		}
		keys, e := FullSlotRouteKeys(dst, level, blocks[pos[0]][pos[1]], r)
		if e != nil {
			return fail(fmt.Errorf("child %v: %w", pos, e))
		}
		slices.Sort(c.Galois)
		if !slices.Equal(keys, c.Galois) {
			return fail(fmt.Errorf("incorrect child %v key inventory", pos))
		}
		for _, k := range keys {
			refs[k]++
		}
		routes[pos[0]][pos[1]] = r
	}
	declared := map[uint64]int{}
	for _, k := range a.Joint.Keys {
		if k.Scope != scope || k.Refs < 1 || declared[k.Galois] != 0 {
			return fail(fmt.Errorf("wrong key scope, refcount or duplicate key"))
		}
		declared[k.Galois] = k.Refs
	}
	if bytesPerKey <= 0 || !maps.Equal(declared, refs) || a.Joint.KeyBytes < 0 || a.Joint.KeyBytes/bytesPerKey != len(refs) || a.Joint.KeyBytes%bytesPerKey != 0 {
		return fail(fmt.Errorf("incorrect joint physical key union or byte charge"))
	}
	return routes, slices.Sorted(maps.Keys(refs)), nil
}
