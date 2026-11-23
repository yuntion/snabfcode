package ringswitch

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/tower"
)

// Engineering fixture, not a public graph conversion or a certificate.
type graphBasis struct {
	SourceN, TargetN, CutLevel, KeyLevelQ, KeyLevelP int
	SourceQ, TargetQ, P                              []uint64
}

func graphParams(t *testing.T) (ckks.Parameters, ckks.Parameters, graphBasis) {
	t.Helper()
	s, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 14, LogQ: []int{50, 40, 40, 40, 40, 40}, LogP: []int{50}, LogDefaultScale: 40})
	if err != nil {
		t.Fatal(err)
	}
	d, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 13, Q: s.Q()[:2], P: s.P(), LogDefaultScale: 40})
	if err != nil {
		t.Fatal(err)
	}
	return s, d, graphBasis{s.N(), d.N(), 2, 1, 0, s.Q(), d.Q(), s.P()}
}

func graphWrite(t *testing.T, name string, value any) {
	t.Helper()
	buf, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(os.Getenv("SNABF_A2B_RUN"), name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.Write(buf); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestA2bGraphBasis(t *testing.T) {
	if os.Getenv("SNABF_A2B_RUN") == "" {
		t.Skip("explicit engineering run only")
	}
	_, _, b := graphParams(t)
	graphWrite(t, "basis.json", b)
}

type graphScreen struct {
	Basis     graphBasis
	Estimator string
	Minimum   map[string]float64
}

func graphGate(b graphBasis, m graphScreen, switchTarget float64) (source, split bool) {
	if !reflect.DeepEqual(b, m.Basis) || m.Estimator != "ae45d740ceb138af87a4f92edc592a0ab0d4f43f" || len(m.Minimum) != 6 || !(switchTarget >= 128) || math.IsInf(switchTarget, 0) {
		return false, false
	}
	pass := func(role string, floor float64) bool {
		x, ok := m.Minimum[role]
		return ok && x >= floor && !math.IsInf(x, 0) && !math.IsNaN(x)
	}
	source = pass("source.Q", 128) && pass("source.QP", 128)
	return source, source && pass("target.Q", 128) && pass("target.QP", 128) && pass("switch.small", switchTarget) && pass("switch.large", switchTarget)
}

func graphRoute(mode string, source, split bool, keepBytes, splitBytes int) string {
	if keepBytes <= 0 || splitBytes <= 0 || !source {
		return "reject"
	}
	switch mode {
	case "keep-source":
		return "keep-source"
	case "force-down":
		if split {
			return "force-down"
		}
	case "adaptive":
		if split && splitBytes < keepBytes {
			return "force-down"
		}
		return "keep-source"
	}
	return "reject"
}

func graphCT(t *testing.T, ct *rlwe.Ciphertext, err error) *rlwe.Ciphertext {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	return ct
}
func graphPairs(x []complex128) [][2]float64 {
	z := make([][2]float64, len(x))
	for i, v := range x {
		z[i] = [2]float64{real(v), imag(v)}
	}
	return z
}

func TestA2bGraphArms(t *testing.T) {
	out, pass1 := os.Getenv("SNABF_A2B_RUN"), os.Getenv("SNABF_TEST_PASS1")
	if out == "" {
		t.Skip("explicit engineering run only")
	}
	if pass1 == "" {
		t.Fatal("native Pass 1 required")
	}
	s, d, b := graphParams(t)
	raw, err := os.ReadFile(filepath.Join(out, "screen.json"))
	if err != nil {
		t.Fatal(err)
	}
	var screen graphScreen
	if err = json.Unmarshal(raw, &screen); err != nil {
		t.Fatal(err)
	}
	sOK, dOK := graphGate(b, screen, 128)
	if !sOK || !dOK {
		t.Fatal("ordinary or switch instance missed 128 bits")
	}
	f := &linearFixture{src: s, dst: d}
	skg, dk := rlwe.NewKeyGenerator(s), rlwe.NewKeyGenerator(d)
	f.sk, f.small = skg.GenSecretKeyNew(), dk.GenSecretKeyNew()
	full, half := s.MaxSlots(), d.MaxSlots()
	width := full
	if os.Getenv("SNABF_A2B_EXTENT") == "tail-one" {
		width = half + 1
	}
	level := b.KeyLevelQ
	kp := rlwe.EvaluationKeyParameters{LevelQ: &level}
	f.shifts = []int{0, 1, half, half + 1, full - 1}
	f.weights = make(tower.Diagonals[float64])
	for j, k := range f.shifts {
		w := make([]float64, full)
		for i := range w {
			if i < width && (i+k)%full < width {
				w[i] = linearWeight(i, j)
			}
		}
		f.weights[k] = w
	}
	blocks, err := tower.PartitionFullSlotDiagonals(f.weights, half, width, width)
	if err != nil {
		t.Fatal(err)
	}
	f.keys.Down = skg.GenEvaluationKeyNew(f.sk, f.small, kp)
	f.keys.Fold = skg.GenGaloisKeyNew(s.GaloisElement(half), f.sk, kp)
	if f.keys.Down.LevelQ() != b.KeyLevelQ || f.keys.Down.LevelP() != b.KeyLevelP {
		t.Fatal("unmeasured switch level")
	}
	wire, err := f.keys.Down.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	var keys []uint64
	f.plan = FullSlotLinearPlan{InputWidth: width, OutputWidth: width, FoldOffsets: [2]int{half, half}}
	f.plan.Routes, keys, _ = jointLinearRoutes(t, pass1, d, blocks, fmt.Sprintf("%x", sha256.Sum256(wire)))
	f.keys.Target = dk.GenGaloisKeysNew(keys, f.small, kp)
	pin, inventory, err := f.keys.inspect(s, d, b.CutLevel)
	if err != nil {
		t.Fatal(err)
	}
	splitEval, err := NewFullSlotLinear(s, d, b.CutLevel, f.weights, f.plan, f.keys, pin)
	if err != nil {
		t.Fatal(err)
	}
	splitBytes := inventory.DownBytes + inventory.FoldBytes + inventory.TargetBytes
	// The full-level +1 key is already required by the prefix and is not charged again.
	gals := []*rlwe.GaloisKey{skg.GenGaloisKeyNew(s.GaloisElement(1), f.sk)}
	keepBytes := 0
	targetLevel := b.CutLevel
	for _, k := range f.shifts[2:] {
		key := skg.GenGaloisKeyNew(s.GaloisElement(k), f.sk, rlwe.EvaluationKeyParameters{LevelQ: &targetLevel})
		v, err := key.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		keepBytes += len(v)
		gals = append(gals, key)
	}
	ev := ckks.NewEvaluator(s, rlwe.NewMemEvaluationKeySet(skg.GenRelinearizationKeyNew(f.sk), gals...))
	route := tower.Route{Level: b.CutLevel, LogSlots: s.LogMaxSlots(), LogN: s.LogN()}
	for _, k := range f.shifts {
		route.Terms = append(route.Terms, [3]int{k, k, 0})
	}
	lt := tower.NewTransformation(s, tower.Parameters{DiagonalsIndexList: f.shifts, LevelQ: b.CutLevel, LevelP: s.MaxLevelP(), Scale: rlwe.NewScale(s.Q()[b.CutLevel]), LogDimensions: s.LogMaxDimensions()}, route)
	if err := tower.Encode(ckks.NewEncoder(s), f.weights, lt); err != nil {
		t.Fatal(err)
	}
	decisions := map[string]string{}
	for _, mode := range []string{"keep-source", "force-down", "adaptive"} {
		decisions[mode] = graphRoute(mode, sOK, dOK, keepBytes, splitBytes)
	}
	s512, d512 := graphGate(b, screen, 512)
	if !s512 || d512 || graphRoute("force-down", s512, d512, keepBytes, splitBytes) != "reject" || graphRoute("adaptive", s512, d512, keepBytes, splitBytes) != "keep-source" {
		t.Fatal("512-bit switching control")
	}
	bad := b
	bad.TargetQ = slices.Clone(b.TargetQ)
	slices.Reverse(bad.TargetQ)
	if a, d := graphGate(bad, screen, 128); a || d {
		t.Fatal("reordered measured basis accepted")
	}
	graphWrite(t, "routing.json", map[string]any{"mode": decisions, "metric": "marginal serialized evaluation key bytes (not latency)", "keep_bytes": keepBytes, "split_bytes": splitBytes, "split_key_id": fmt.Sprintf("%x", pin), "switch_512_force": "reject", "switch_512_adaptive": "keep-source"})
	// Decisions are frozen before any encryption, decryption or observed error.
	for trial := 0; trial < 2; trial++ {
		x := make([]complex128, full)
		for i := range x {
			if i < width {
				x[i] = complex(float64((37*i+17*trial+13*(i/half))%251-125)/256, 0)
			}
		}
		ct := f.encrypt(t, x)
		must := func(c *rlwe.Ciphertext, e error) *rlwe.Ciphertext { return graphCT(t, c, e) }
		r := must(ev.RotateNew(ct, 1))
		u := must(ev.AddNew(ct, r))
		u = must(ev.MulNew(u, 0.125))
		if err = ev.Rescale(u, u); err != nil {
			t.Fatal(err)
		}
		v := must(ev.MulNew(u, 0.25))
		if err = ev.Rescale(v, v); err != nil {
			t.Fatal(err)
		}
		v = must(ev.AddNew(v, 0.5))
		u = u.CopyNew()
		ev.DropLevel(u, u.Level()-v.Level())
		cut := must(ev.MulRelinNew(u, v))
		if err = ev.Rescale(cut, cut); err != nil {
			t.Fatal(err)
		}
		if cut.Level() != b.CutLevel {
			t.Fatal("wrong graph cut level")
		}
		before := cut.CopyNew()
		wire, err := cut.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		cutScale, _ := cut.Scale.Value.Rat(nil)
		graphWrite(t, fmt.Sprintf("trial%d-input.json", trial), map[string]any{"width": width, "cut_scale": cutScale.RatString(), "x": graphPairs(x), "cut": graphPairs(splitDecode(t, s, f.sk, cut)), "cut_sha256": fmt.Sprintf("%x", sha256.Sum256(wire))})
		for _, mode := range []string{"keep-source", "force-down", "adaptive"} {
			var y []complex128
			var obs any
			switch decisions[mode] {
			case "keep-source":
				z, counts, e := tower.NewEvaluator(ev).EvaluateCounted(cut, lt)
				if e != nil {
					t.Fatal(e)
				}
				if e = ev.Rescale(z, z); e != nil {
					t.Fatal(e)
				}
				if z.Level() != 1 {
					t.Fatal("wrong keep level")
				}
				y = splitDecode(t, s, f.sk, z)
				obs = counts
			case "force-down":
				parts, counts, e := splitEval.Eval(cut)
				if e != nil {
					t.Fatal(e)
				}
				var wires [2]SlotBlock[[]complex128]
				for i, p := range parts {
					if p.Value.Level() != 0 {
						t.Fatal("wrong split level")
					}
					wires[1-i] = SlotBlock[[]complex128]{p.Start, p.Used, splitDecode(t, d, f.small, p.Value)}
				}
				logical, e := ReconstructFullSlot(half, width, wires)
				if e != nil {
					t.Fatal(e)
				}
				if counts.CompletedBlocks != 4 || counts.CrossBlockAdds != 2 || counts.Split.DownConvCount != 2 {
					t.Fatal("incomplete forced suffix")
				}
				y = make([]complex128, full)
				for _, block := range wires {
					copy(y[block.Start:], block.Value)
				}
				if !slices.Equal(logical, y[:width]) {
					t.Fatal("logical output differs from physical slot order")
				}
				obs = counts
			default:
				t.Fatal("no legal engineering arm")
			}
			if !cut.Equal(before) || len(y) != full {
				t.Fatal("shared input modified or short output")
			}
			graphWrite(t, fmt.Sprintf("trial%d-%s.json", trial, mode), map[string]any{"selected": decisions[mode], "cut_sha256": fmt.Sprintf("%x", sha256.Sum256(wire)), "observed": obs, "slots": graphPairs(y)})
		}
	}
}
