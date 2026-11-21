package ringswitch

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"slices"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/tower"
)

// Same global weights as the original private partition / joint-Pass-1 probe.
// The oracle below never accesses a partition, a route, or a packed output.
func linearWeight(row, m int) float64 {
	return [...]float64{.25, -.5, .125, .375, -.0625}[m] * float64(1+(row+3*m)%5) / 4
}

type linearFixture struct {
	src, dst  ckks.Parameters
	sk, small *rlwe.SecretKey
	shifts    []int
	weights   tower.Diagonals[float64]
	plan      FullSlotLinearPlan
	keys      FullSlotKeys
	pin       [32]byte
	downID    string
	inventory FullSlotKeyInventory
}

func newLinearFixture(t *testing.T, logN int, planner string) *linearFixture {
	t.Helper()
	f := new(linearFixture)
	f.src, f.dst = splitParams(t, logN)
	s, full, l := f.dst.MaxSlots(), f.src.MaxSlots(), f.dst.MaxLevel()
	f.plan = FullSlotLinearPlan{InputWidth: full, OutputWidth: full, FoldOffsets: [2]int{s, s}}
	f.shifts = []int{0, 1, s, s + 1, full - 1}
	f.weights = make(tower.Diagonals[float64])
	for m, k := range f.shifts {
		d := make([]float64, full)
		for i := range d {
			d[i] = linearWeight(i, m)
		}
		f.weights[k] = d
	}
	blocks, err := tower.PartitionFullSlotDiagonals(f.weights, s, full, full)
	if err != nil {
		t.Fatal(err)
	}
	large, small := rlwe.NewKeyGenerator(f.src), rlwe.NewKeyGenerator(f.dst)
	f.sk, f.small = large.GenSecretKeyNew(), small.GenSecretKeyNew()
	kp := rlwe.EvaluationKeyParameters{LevelQ: &l}
	f.keys.Down = large.GenEvaluationKeyNew(f.sk, f.small, kp)
	f.keys.Fold = large.GenGaloisKeyNew(f.src.GaloisElement(s), f.sk, kp)
	wire, err := f.keys.Down.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	f.downID = fmt.Sprintf("%x", sha256.Sum256(wire))
	f.inventory.DownBytes = len(wire)
	wire, err = f.keys.Fold.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	f.inventory.FoldBytes = len(wire)
	var keys []uint64
	plannedBytes := -1
	if planner != "" {
		f.plan.Routes, keys, plannedBytes = jointLinearRoutes(t, planner, f.dst, blocks, f.downID)
	} else {
		keys = []uint64{f.dst.GaloisElement(1), f.dst.GaloisElement(s - 1)}
		for o, row := range blocks {
			for i, child := range row {
				r := &tower.Route{Level: l, LogSlots: f.dst.LogMaxSlots(), LogN: f.dst.LogN()}
				for _, d := range slices.Sorted(maps.Keys(child)) {
					term := [3]int{d, d, 0}
					if d == s-1 {
						term = [3]int{d, 0, d}
					}
					r.Terms = append(r.Terms, term)
				}
				f.plan.Routes[o][i] = r
			}
		}
	}
	f.keys.Target = small.GenGaloisKeysNew(keys, f.small, kp)
	for _, k := range f.keys.Target {
		wire, err := k.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		f.inventory.TargetKeys++
		f.inventory.TargetBytes += len(wire)
	}
	if plannedBytes >= 0 && f.inventory.TargetBytes != plannedBytes {
		t.Fatalf("joint priced %d bytes, actual key bytes %d", plannedBytes, f.inventory.TargetBytes)
	}
	f.pin, err = f.keys.Commitment(f.src, f.dst, f.src.MaxLevel())
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func (f *linearFixture) encrypt(t *testing.T, x []complex128) *rlwe.Ciphertext {
	t.Helper()
	p := ckks.NewPlaintext(f.src, f.src.MaxLevel())
	if err := ckks.NewEncoder(f.src).Encode(x, p); err != nil {
		t.Fatal(err)
	}
	ct, err := rlwe.NewEncryptor(f.src, f.sk).EncryptNew(p)
	if err != nil {
		t.Fatal(err)
	}
	return ct
}

// Independent expected counts from the algebra of the selected route. The
// runtime observations are incremented at the actual successful evaluator calls.
func expectedCounts(r *tower.Route) tower.TransformCounts {
	baby, group := map[int]bool{}, map[int]bool{}
	for _, term := range r.Terms {
		if term[1] != 0 {
			baby[term[1]] = true
		}
		group[term[2]] = true
	}
	giants := len(group)
	if group[0] {
		giants--
	}
	return tower.TransformCounts{BabyRotations: len(baby), GiantRotations: giants, PlainProducts: len(r.Terms), FusedAdds: len(r.Terms) - len(group), GroupAdds: len(group) - 1}
}

func runFullSlotLinear(t *testing.T, logN, trials int, planner string) {
	t.Helper()
	f := newLinearFixture(t, logN, planner)
	e, err := NewFullSlotLinear(f.src, f.dst, f.src.MaxLevel(), f.weights, f.plan, f.keys, f.pin)
	if err != nil {
		t.Fatal(err)
	}
	s, full := f.dst.MaxSlots(), f.src.MaxSlots()
	rng := rand.New(rand.NewSource(20261121))
	threshold := math.Exp2(-12)
	var last FullSlotLinearObservation
	var worstTrial, worstSlot int
	peaks := make([]float64, trials)
	peak := 0.0
	for tr := range trials {
		x := make([]complex128, full)
		if tr != 0 {
			for i := range x {
				x[i] = complex(rng.Float64()-.5, rng.Float64()-.5)
			}
		}
		x[0], x[s-1], x[s], x[full-1] = .75, .125+.375i, -.25, -.25+.125i
		ct := f.encrypt(t, x)
		before := ct.CopyNew()
		out, r, err := e.Eval(ct)
		if err != nil {
			t.Fatal(err)
		}
		last = r
		z := r.Split
		if !ct.Equal(before) || out[0].Value == out[1].Value || r.Keys != f.inventory || r.CompletedBlocks != 4 || r.CrossBlockAdds != 2 || r.TargetRescales != 2 || z.MaskMulCount != 2 || z.FoldRotCount != 2 || z.FoldAddCount != 2 || z.RescaleCount != 2 || z.DownConvCount != 2 || z.SwitchKeyBytes != f.inventory.DownBytes || z.FoldGaloisElement != f.src.GaloisElement(s) {
			t.Fatalf("alias or execution count mismatch: %+v", r)
		}
		var parts [2]SlotBlock[[]complex128]
		for o, b := range out {
			sp := z.Outputs[o]
			if b.Start != o*s || b.Value.Level() != f.src.MaxLevel()-2 || b.Value.Value[0].N() != f.dst.N() || b.Value.LogDimensions != f.dst.LogMaxDimensions() || math.Abs(b.Value.LogScale()-ct.LogScale()) > 1e-8 || sp.N != f.dst.N() || sp.Level != ct.Level()-1 || sp.LogicalStart != o*s || sp.UsedSlots != s || sp.DownKeySHA256 != f.downID {
				t.Fatalf("unexpected state at sink %d", o)
			}
			for i, route := range f.plan.Routes[o] {
				if want := expectedCounts(route); r.Children[o][i] != want {
					t.Fatalf("actual child [%d,%d] %+v; expected %+v", o, i, r.Children[o][i], want)
				}
			}
			parts[1-o] = SlotBlock[[]complex128]{b.Start, b.Used, splitDecode(t, f.dst, f.small, b.Value)} // reverse delivery
		}
		got, err := ReconstructFullSlot(s, full, parts)
		if err != nil {
			t.Fatal(err)
		}
		for i, y := range got {
			var want complex128
			for m, k := range f.shifts {
				want += complex(linearWeight(i, m), 0) * x[(i+k)%full]
			}
			d := cmplx.Abs(y - want)
			if !(d < threshold) {
				t.Fatalf("trial %d slot %d: got %v want %v error=%g", tr, i, y, want, d)
			}
			peaks[tr] = max(peaks[tr], d)
			if d > peak {
				peak, worstTrial, worstSlot = d, tr, i
			}
		}
	}
	t.Logf("N=%d->%d L=%d->%d->%d trials=%d checked=%d Linf=%.17g at trial=%d global_slot=%d keys=%+v; per-input children=%+v", f.src.N(), f.dst.N(), f.src.MaxLevel(), f.src.MaxLevel()-1, f.src.MaxLevel()-2, trials, trials*full, peak, worstTrial, worstSlot, last.Keys, last.Children)
	if planner != "" {
		data, err := json.MarshalIndent(map[string]any{"scope": "runtime only: one split and one target linear, not a compiled graph arm", "source_n": f.src.N(), "target_n": f.dst.N(), "source_q": f.src.Q(), "target_q": f.dst.Q(), "p": f.src.P(), "input_level": f.src.MaxLevel(), "output_level": f.src.MaxLevel() - 2, "seed": 20261121, "trials": trials, "slots_per_trial": full, "checked_slots": trials * full, "threshold": threshold, "max_error": peak, "worst_trial": worstTrial, "worst_global_slot": worstSlot, "per_trial_max": peaks, "per_input_observation": last, "provisioned_key_commitment": fmt.Sprintf("%x", f.pin)}, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		linearEvidence(t, "observations.json", data)
	}
}

func TestFullSlotLinearRuntime(t *testing.T) { runFullSlotLinear(t, 10, 3, "") }
func TestFullSlotLinearJoint(t *testing.T) {
	exe := os.Getenv("SNABF_TEST_PASS1")
	if exe == "" {
		t.Skip("set SNABF_TEST_PASS1 to the native planner; the release gate requires it")
	}
	runFullSlotLinear(t, 14, 64, exe)
}
