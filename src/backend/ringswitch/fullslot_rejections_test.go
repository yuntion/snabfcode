package ringswitch

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/tower"
)

func TestFullSlotLinearRejects(t *testing.T) {
	f := newLinearFixture(t, 10, "")
	reject := func(t *testing.T, p FullSlotLinearPlan, k FullSlotKeys, pin [32]byte, want string) {
		t.Helper()
		e, err := NewFullSlotLinear(f.src, f.dst, f.src.MaxLevel(), f.weights, p, k, pin)
		if err == nil || e != nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("expected %q before evaluation; got %v", want, err)
		}
		t.Log(err)
	}
	for o := range 2 {
		for i := range 2 {
			t.Run(fmt.Sprintf("missing-block-%d%d", o, i), func(t *testing.T) {
				p := f.plan
				p.Routes[o][i] = nil
				reject(t, p, f.keys, f.pin, fmt.Sprintf("block [%d,%d]", o, i))
			})
		}
		t.Run(fmt.Sprintf("missing-fold-%d", o), func(t *testing.T) {
			p := f.plan
			p.FoldOffsets[o] = 0
			reject(t, p, f.keys, f.pin, "both source half-period folds")
		})
	}
	large, small := rlwe.NewKeyGenerator(f.src), rlwe.NewKeyGenerator(f.dst)
	l := f.dst.MaxLevel()
	kp := rlwe.EvaluationKeyParameters{LevelQ: &l}
	otherSrc, otherDst := large.GenSecretKeyNew(), small.GenSecretKeyNew()
	for _, tc := range []struct {
		name    string
		replace func(*FullSlotKeys)
	}{
		{"wrong-fold-secret", func(k *FullSlotKeys) { k.Fold = large.GenGaloisKeyNew(f.keys.Fold.GaloisElement, otherSrc, kp) }},
		{"wrong-switch-destination", func(k *FullSlotKeys) { k.Down = large.GenEvaluationKeyNew(f.sk, otherDst, kp) }},
		{"wrong-target-rotation-secret", func(k *FullSlotKeys) {
			k.Target = slices.Clone(k.Target)
			k.Target[0] = small.GenGaloisKeyNew(k.Target[0].GaloisElement, otherDst, kp)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			k := f.keys
			tc.replace(&k)
			// Check that shape alone would accept the replacement. The unchanged
			// pin came from the real source/target secrets, before this substitution.
			other, err := k.Commitment(f.src, f.dst, f.src.MaxLevel())
			if err != nil || other == f.pin {
				t.Fatalf("invalid substitution: %v", err)
			}
			reject(t, f.plan, k, f.pin, "provisioned full-slot key commitment")
		})
	}
	t.Run("missing-required-target-key", func(t *testing.T) {
		k := f.keys
		k.Target = k.Target[:1]
		p, err := k.Commitment(f.src, f.dst, f.src.MaxLevel())
		if err != nil {
			t.Fatal(err)
		}
		reject(t, f.plan, k, p, "target key inventory")
	})
	t.Run("unfolded-input", func(t *testing.T) {
		e, err := NewFullSlotLinear(f.src, f.dst, f.src.MaxLevel(), f.weights, f.plan, f.keys, f.pin)
		if err != nil {
			t.Fatal(err)
		}
		out, r, err := e.Eval(ckks.NewCiphertext(f.dst, 1, l))
		if err == nil || out[0].Value != nil || out[1].Value != nil || r.Split.DownConvCount != 0 || r.CompletedBlocks != 0 {
			t.Fatalf("returned partial output or accepted an unchecked target input: %+v", r)
		}
	})
}

// Raw Lattigo counterfactual (not an admitted evaluator path): masking and
// switching without the half-period rotate+add divides each half by two.
func TestFullSlotLinearOmittedFoldWitness(t *testing.T) {
	f := newLinearFixture(t, 10, "")
	s := f.dst.MaxSlots()
	e, err := NewFullSlotSplitter(f.src, f.dst, f.src.MaxLevel(), 2*s, f.keys.Fold, f.keys.Down)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]complex128, 2*s)
	x[0], x[s] = .75, -.25
	ct := f.encrypt(t, x)
	correct, _, err := e.Split(ct)
	if err != nil {
		t.Fatal(err)
	}
	for h := range 2 {
		masked, err := e.eval.MulNew(ct, e.masks[h])
		if err != nil {
			t.Fatal(err)
		}
		if err = e.eval.Rescale(masked, masked); err != nil {
			t.Fatal(err)
		}
		if _, err = e.projectFolded(foldedHalf{ct: masked}); err == nil {
			t.Fatal("projection admitted a mask without its fold producer")
		}
		raw := ckks.NewCiphertext(f.dst, 1, masked.Level())
		if err = e.eval.ApplyEvaluationKey(masked, f.keys.Down, raw); err != nil {
			t.Fatal(err)
		}
		raw.LogDimensions = f.dst.LogMaxDimensions()
		bad, good := splitDecode(t, f.dst, f.small, raw), splitDecode(t, f.dst, f.small, correct[h])
		for j := range s {
			splitClose(t, bad[j], x[h*s+j]/2)
			splitClose(t, good[j], x[h*s+j])
		}
		t.Logf("half=%d global=%d wanted=%g without_fold=%.9f checked_fold=%.9f", h, h*s, real(x[h*s]), real(bad[0]), real(good[0]))
	}
}

func TestFullSlotLinearZeroQuadrantsAndTail(t *testing.T) {
	f := newLinearFixture(t, 10, "")
	s := f.dst.MaxSlots()
	p := f.plan
	p.InputWidth, p.OutputWidth = s+3, s+1
	p.Routes = tower.FullSlotRoutes{}
	w := tower.Diagonals[float64]{0: make([]float64, 2*s)}
	for j := range p.OutputWidth {
		w[0][j] = 1
	}
	for h := range 2 {
		p.Routes[h][h] = &tower.Route{Level: f.dst.MaxLevel(), LogSlots: f.dst.LogMaxSlots(), LogN: f.dst.LogN(), Terms: [][3]int{{0, 0, 0}}}
	}
	k := f.keys
	k.Target = nil
	pin, err := k.Commitment(f.src, f.dst, f.src.MaxLevel())
	if err != nil {
		t.Fatal(err)
	}
	bad := p
	bad.Routes[0][1] = p.Routes[0][0]
	if e, err := NewFullSlotLinear(f.src, f.dst, f.src.MaxLevel(), w, bad, k, pin); e != nil || err == nil {
		t.Fatal("surplus route in a zero quadrant accepted")
	}
	e, err := NewFullSlotLinear(f.src, f.dst, f.src.MaxLevel(), w, p, k, pin)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]complex128, 2*s)
	for j := range x {
		x[j] = complex(float64(j)/(2*float64(s)), .25)
	}
	out, r, err := e.Eval(f.encrypt(t, x))
	if err != nil {
		t.Fatal(err)
	}
	if r.CompletedBlocks != 2 || r.CrossBlockAdds != 0 || r.TargetRescales != 2 || r.Keys.TargetKeys != 0 || r.Children[0][1] != (tower.TransformCounts{}) || r.Children[1][0] != (tower.TransformCounts{}) {
		t.Fatal(r)
	}
	var parts [2]SlotBlock[[]complex128]
	for h, b := range out {
		v := splitDecode(t, f.dst, f.small, b.Value)
		parts[1-h] = SlotBlock[[]complex128]{b.Start, b.Used, v}
		for j, y := range v {
			var want complex128
			if h*s+j < p.OutputWidth {
				want = x[h*s+j]
			}
			splitClose(t, y, want)
		}
	}
	got, err := ReconstructFullSlot(s, p.OutputWidth, parts)
	if err != nil || len(got) != p.OutputWidth {
		t.Fatal(len(got), err)
	}
	for j, y := range got {
		splitClose(t, y, x[j])
	}
	for _, tc := range []struct {
		name    string
		corrupt func(*[2]SlotBlock[[]complex128])
	}{
		{"duplicate-start", func(b *[2]SlotBlock[[]complex128]) { b[0] = b[1] }},
		{"missing-block", func(b *[2]SlotBlock[[]complex128]) { b[0] = SlotBlock[[]complex128]{} }},
		{"bad-start", func(b *[2]SlotBlock[[]complex128]) { b[0].Start = -1 }},
		{"short-decode", func(b *[2]SlotBlock[[]complex128]) { b[0].Value = b[0].Value[:1] }},
		{"wrong-tail-extent", func(b *[2]SlotBlock[[]complex128]) { b[0].Used = s }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := parts
			tc.corrupt(&b)
			if v, err := ReconstructFullSlot(s, p.OutputWidth, b); v != nil || err == nil {
				t.Fatal("malformed block reconstruction accepted")
			}
		})
	}
	if v, err := ReconstructFullSlot(math.MaxInt, 2, parts); v != nil || err == nil {
		t.Fatal("bad geometry accepted")
	}
}
