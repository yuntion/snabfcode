// Package tower executes the exact subgroup/coset split selected by Pass 1.
package tower

import (
	"fmt"
	native "github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"slices"
	"snabf.local/compiler/src/backend/gadget"
)

type Parameters = native.Parameters
type Diagonals[T ckks.Float] = native.Diagonals[T]

type Route struct {
	Level, LogSlots, LogN int
	Terms                 [][3]int            // (physical diagonal lift, baby, giant)
	Babies, Giants        map[int]gadget.Site // nil for frozen Pass 1/2
}

type product struct {
	diagonal, baby int
	mask           *rlwe.Plaintext
}
type group struct {
	giant int
	terms []product
}
type LinearTransformation struct {
	params Parameters
	route  Route
	babies []int
	groups []group
}

func NewTransformation(params rlwe.ParameterProvider, p Parameters, r Route) LinearTransformation {
	physical := params.GetRLWEParameters().N() / 2
	if 1<<r.LogN != 2*physical || p.LevelQ != r.Level || p.LogDimensions.Rows != 0 || r.LogSlots != p.LogDimensions.Cols || r.LogSlots >= r.LogN || len(r.Terms) == 0 {
		panic("Pass 1 plan does not match the HEIR transform budget/layout")
	}
	mask := (1 << r.LogSlots) - 1
	t := LinearTransformation{params: p, route: r}
	want := map[int]bool{}
	for _, d := range p.DiagonalsIndexList {
		want[d&mask] = true
	}
	if len(want) != len(r.Terms) || len(want) != len(p.DiagonalsIndexList) {
		panic("Pass 1 diagonal count mismatch")
	}
	byGiant := map[int][]product{}
	babySet := map[int]bool{}
	for _, term := range r.Terms {
		d, b, g := term[0], term[1], term[2]
		if !want[d] || b < 0 || g < 0 || b >= physical || g >= physical || (b+g)%physical != d {
			panic("Pass 1 invalid physical decomposition")
		}
		delete(want, d)
		if b != 0 {
			babySet[b] = true
		}
		byGiant[g] = append(byGiant[g], product{diagonal: d, baby: b})
	}
	for b := range babySet {
		t.babies = append(t.babies, b)
	}
	slices.Sort(t.babies)
	for g, terms := range byGiant {
		t.groups = append(t.groups, group{g, terms})
	}
	slices.SortFunc(t.groups, func(a, b group) int { return a.giant - b.giant })
	return t
}

func Encode[T float64](enc *ckks.Encoder, diags Diagonals[T], t LinearTransformation) error {
	slots := 1 << t.route.LogSlots
	packed := map[int][]T{}
	for d, v := range diags {
		packed[d&(slots-1)] = v
	}
	for _, coset := range t.groups {
		for i := range coset.terms {
			term := &coset.terms[i]
			source := packed[term.diagonal]
			if len(source) != slots {
				return fmt.Errorf("invalid HEIR mask %d length %d", term.diagonal, len(source))
			}
			values := make([]float64, slots)
			// Rot_g(Mul(Rot_-g(mask_d), Rot_b(x))) with d=b+g mod N/2.
			for j := range values {
				values[j] = float64(source[(j-coset.giant)&(slots-1)])
			}
			pt := ckks.NewPlaintext(enc.GetParameters(), t.params.LevelQ)
			pt.LogDimensions = t.params.LogDimensions
			pt.Scale = t.params.Scale
			if err := enc.Encode(values, pt); err != nil {
				return err
			}
			term.mask = pt
		}
	}
	return nil
}

type Evaluator struct{ eval *ckks.Evaluator }

func NewEvaluator(e *ckks.Evaluator) Evaluator { return Evaluator{e} }

// Counts completed operations, not distinct rotation keys. PlainProducts
// includes MulThenAdd; FusedAdds counts the additions inside those products.
type TransformCounts struct {
	BabyRotations, GiantRotations, PlainProducts, FusedAdds, GroupAdds int
}

func (e Evaluator) EvaluateNew(ct *rlwe.Ciphertext, t LinearTransformation) (*rlwe.Ciphertext, error) {
	var counts TransformCounts
	return e.evaluate(ct, t, &counts)
}

func (e Evaluator) EvaluateCounted(ct *rlwe.Ciphertext, t LinearTransformation) (*rlwe.Ciphertext, TransformCounts, error) {
	var counts TransformCounts
	out, err := e.evaluate(ct, t, &counts)
	return out, counts, err
}

func (e Evaluator) evaluate(ct *rlwe.Ciphertext, t LinearTransformation, counts *TransformCounts) (*rlwe.Ciphertext, error) {
	if ct.Degree() != 1 || ct.Level() != t.params.LevelQ || !ct.IsBatched || ct.LogDimensions != t.params.LogDimensions || ct.Value[0].N() != 1<<t.route.LogN {
		return nil, fmt.Errorf("Pass 1 ciphertext does not match prepared transform")
	}
	rot := map[int]*rlwe.Ciphertext{}
	var err error
	if len(t.babies) > 0 {
		if t.route.Babies != nil {
			rot, err = gadget.Babies(e.eval, ct, t.babies, t.route.Babies)
		} else {
			rot, err = e.eval.RotateHoistedNew(ct, t.babies)
		}
		if err != nil {
			return nil, err
		}
		counts.BabyRotations += len(rot)
	}
	rot[0] = ct
	var result *rlwe.Ciphertext
	for _, coset := range t.groups {
		var sum *rlwe.Ciphertext
		for _, term := range coset.terms {
			if term.mask == nil {
				return nil, fmt.Errorf("unencoded Pass 1 mask")
			}
			first := sum == nil
			if first {
				sum, err = e.eval.MulNew(rot[term.baby], term.mask)
			} else {
				err = e.eval.MulThenAdd(rot[term.baby], term.mask, sum)
			}
			if err != nil {
				return nil, err
			}
			counts.PlainProducts++
			if !first {
				counts.FusedAdds++
			}
		}
		if coset.giant != 0 {
			if t.route.Giants != nil {
				s, ok := t.route.Giants[coset.giant]
				if !ok {
					return nil, fmt.Errorf("missing giant plan")
				}
				sum, err = gadget.Giant(e.eval, sum, coset.giant, s)
			} else {
				err = e.eval.Rotate(sum, coset.giant, sum)
			}
			if err != nil {
				return nil, err
			}
			counts.GiantRotations++
		}
		if result == nil {
			result = sum
		} else if err = e.eval.Add(result, sum, result); err != nil {
			return nil, err
		} else {
			counts.GroupAdds++
		}
	}
	return result, nil // HEIR owns the following rescale, including its output buffer.
}
