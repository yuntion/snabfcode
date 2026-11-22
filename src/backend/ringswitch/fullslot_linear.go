package ringswitch

import (
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/tower"
)

type FullSlotLinearPlan struct {
	InputWidth, OutputWidth int
	FoldOffsets             [2]int
	Routes                  tower.FullSlotRoutes
}

// A prepared serial runtime for one split followed by one linear map. Holds
// only evaluation keys, encoded public weights and evaluator scratch space.
type FullSlotLinear struct {
	split       *FullSlotSplitter
	target      *ckks.Evaluator
	children    [2][2]*tower.LinearTransformation
	keys        FullSlotKeyInventory
	outputWidth int
}

// weights are the source matrix, not four caller-supplied child masks. Only
// mathematically empty quadrants may omit a route. This slice requires at
// least one nonzero quadrant per output half; it does not synthesize zeros.
func NewFullSlotLinear(src, dst ckks.Parameters, sourceLevel int, weights tower.Diagonals[float64], plan FullSlotLinearPlan, keys FullSlotKeys, pinned [32]byte) (*FullSlotLinear, error) {
	s, l := dst.MaxSlots(), sourceLevel-1
	if src.N() != 2*dst.N() || l < 1 || l > dst.MaxLevel() || sourceLevel > src.MaxLevel() || dst.LevelsConsumedPerRescaling() != 1 || plan.OutputWidth <= s {
		return nil, fmt.Errorf("expected two occupied output halves and one Q for each of split and linear rescale")
	}
	if plan.FoldOffsets != [2]int{s, s} {
		return nil, fmt.Errorf("both source half-period folds are required")
	}
	blocks, err := tower.PartitionFullSlotDiagonals(weights, s, plan.InputWidth, plan.OutputWidth)
	if err != nil {
		return nil, err
	}
	required := map[uint64]bool{}
	for o, row := range blocks {
		if len(row[0])+len(row[1]) == 0 {
			return nil, fmt.Errorf("unsupported empty output half %d", o)
		}
		for i, b := range row {
			gs, err := tower.FullSlotRouteKeys(dst, l, b, plan.Routes[o][i])
			if err != nil {
				return nil, fmt.Errorf("block [%d,%d]: %w", o, i, err)
			}
			for _, g := range gs {
				required[g] = true
			}
		}
	}
	id, inventory, err := keys.inspect(src, dst, sourceLevel)
	if err != nil {
		return nil, err
	}
	if id != pinned {
		return nil, fmt.Errorf("evaluation keys do not match provisioned full-slot key commitment")
	}
	if len(required) != len(keys.Target) {
		return nil, fmt.Errorf("target key inventory does not cover selected routes exactly")
	}
	bound := make([]*rlwe.GaloisKey, 0, len(keys.Target))
	for _, k := range keys.Target {
		if !required[k.GaloisElement] {
			return nil, fmt.Errorf("unexpected target rotation key %d", k.GaloisElement)
		}
		bound = append(bound, k.CopyNew())
	}
	f, err := NewFullSlotSplitter(src, dst, sourceLevel, plan.InputWidth, keys.Fold, keys.Down)
	if err != nil {
		return nil, err
	}
	e := &FullSlotLinear{split: f, target: ckks.NewEvaluator(dst, rlwe.NewMemEvaluationKeySet(nil, bound...)), keys: inventory, outputWidth: plan.OutputWidth}
	enc := ckks.NewEncoder(dst)
	for o, row := range blocks {
		for i, b := range row {
			if len(b) == 0 {
				continue
			}
			p := tower.Parameters{DiagonalsIndexList: slices.Sorted(maps.Keys(b)), LevelQ: l, LevelP: dst.MaxLevelP(), Scale: rlwe.NewScale(dst.Q()[l]), LogDimensions: dst.LogMaxDimensions()}
			x := tower.NewTransformation(dst, p, *plan.Routes[o][i])
			if err := tower.Encode(enc, b, x); err != nil {
				return nil, fmt.Errorf("encode block [%d,%d]: %w", o, i, err)
			}
			e.children[o][i] = &x
		}
	}
	return e, nil
}

type FullSlotLinearObservation struct {
	Split                                           SplitObservation
	Keys                                            FullSlotKeyInventory
	Children                                        [2][2]tower.TransformCounts
	CompletedBlocks, CrossBlockAdds, TargetRescales int
}

// Start is in logical slots of the source-width vector, not a target rotation.
type SlotBlock[T any] struct {
	Start, Used int
	Value       T
}

// Eval does not accept raw target ciphertexts as a substitute for checked
// fold/projection producers. On failure no (even partially filled) outputs
// escape; counts still describe calls completed before the error.
func (e *FullSlotLinear) Eval(in *rlwe.Ciphertext) ([2]SlotBlock[*rlwe.Ciphertext], FullSlotLinearObservation, error) {
	var out, empty [2]SlotBlock[*rlwe.Ciphertext]
	parts, split, err := e.split.Split(in)
	r := FullSlotLinearObservation{Split: split, Keys: e.keys}
	if err != nil {
		return empty, r, err
	}
	eval := tower.NewEvaluator(e.target)
	for o, row := range e.children {
		var sum *rlwe.Ciphertext
		for i, x := range row {
			if x == nil {
				continue
			}
			piece, counts, err := eval.EvaluateCounted(parts[i], *x)
			r.Children[o][i] = counts
			if err != nil {
				return empty, r, fmt.Errorf("evaluate block [%d,%d]: %w", o, i, err)
			}
			r.CompletedBlocks++
			if sum == nil {
				sum = piece
				continue
			}
			if sum.Level() != piece.Level() || !sum.Scale.Equal(piece.Scale) {
				return empty, r, fmt.Errorf("incompatible target sum %d", o)
			}
			if err := e.target.Add(sum, piece, sum); err != nil {
				return empty, r, err
			}
			r.CrossBlockAdds++
		}
		if err := e.target.Rescale(sum, sum); err != nil {
			return empty, r, err
		}
		r.TargetRescales++
		s := e.split.dst.MaxSlots()
		out[o] = SlotBlock[*rlwe.Ciphertext]{o * s, min(s, e.outputWidth-o*s), sum}
	}
	return out, r, nil
}

// ReconstructFullSlot is a post-decode operation. Each block must supply all
// physical slots from its target ciphertext, including the final padded tail.
func ReconstructFullSlot[T float64 | complex128](childSlots, outputWidth int, parts [2]SlotBlock[[]T]) ([]T, error) {
	if childSlots < 1 || childSlots&(childSlots-1) != 0 || childSlots > math.MaxInt/2 || outputWidth <= childSlots || outputWidth > 2*childSlots {
		return nil, fmt.Errorf("invalid two-block output extent")
	}
	seen := [2]bool{}
	for _, b := range parts {
		if (b.Start != 0 && b.Start != childSlots) || b.Used != min(childSlots, outputWidth-b.Start) || len(b.Value) != childSlots || seen[b.Start/childSlots] {
			return nil, fmt.Errorf("missing, duplicate or malformed decoded block at %d", b.Start)
		}
		seen[b.Start/childSlots] = true
	}
	out := make([]T, outputWidth)
	for _, b := range parts {
		copy(out[b.Start:], b.Value)
	}
	return out, nil
}
