package pass1_test

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"regexp"
	"slices"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	hc "snabf.local/compiler/build/pass1/control/heir"
	ic "snabf.local/compiler/build/pass1/control/independent"
	jc "snabf.local/compiler/build/pass1/control/joint"
	hr "snabf.local/compiler/build/pass1/reuse64/heir"
	ir "snabf.local/compiler/build/pass1/reuse64/independent"
	jr "snabf.local/compiler/build/pass1/reuse64/joint"
	hs "snabf.local/compiler/build/pass1/strided64/heir"
	is "snabf.local/compiler/build/pass1/strided64/independent"
	js "snabf.local/compiler/build/pass1/strided64/joint"
)

const tolerance = 1.0 / (1 << 12)

type arm struct {
	keys     []uint64
	keyBytes int
	run      func([]float32) ([]float32, int, int, float64, time.Duration)
}
type configFn func() (*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor)
type encryptFn func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, []float32) []*rlwe.Ciphertext
type decryptFn func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Decryptor, []*rlwe.Ciphertext) []float32

func newArm[T any](t *testing.T, cfg configFn, prep func(ckks.Parameters, *ckks.Encoder) ([]T, []*rlwe.Plaintext),
	infer func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, []*rlwe.Ciphertext, []T, []*rlwe.Plaintext) []*rlwe.Ciphertext,
	encrypt encryptFn, decrypt decryptFn) arm {
	t.Helper()
	eval, p, encoder, enc, dec := cfg()
	lt, pt := prep(p, encoder)
	if len(lt) != 2 {
		t.Fatalf("expected 2 transforms, got %d", len(lt))
	}
	keys := eval.GetGaloisKeysList()
	slices.Sort(keys)
	total := 0
	for _, g := range keys {
		k, err := eval.GetGaloisKey(g)
		if err != nil {
			t.Fatal(err)
		}
		serialized, err := k.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		total += len(serialized)
	}
	return arm{keys, total, func(x []float32) ([]float32, int, int, float64, time.Duration) {
		ct := encrypt(eval, p, encoder, enc, x)
		level := ct[0].Level()
		start := time.Now()
		y := infer(eval, p, encoder, ct, lt, pt)
		dt := time.Since(start)
		return decrypt(eval, p, encoder, dec, y), level, y[0].Level(), y[0].LogScale(), dt
	}}
}

type planArm struct {
	KeyBytes int `json:"key_bytes"`
	Keys     []struct {
		Galois uint64
		Refs   int
	}
	Routes []struct {
		S     int
		Terms [][3]int
	}
}
type plan struct {
	Independent, Joint planArm
	Regions            []struct {
		Diagonals []int
		Level     int
	}
	Bootstrap struct{ Status string }
}

// Read the original dense linalg weights, before packing or HEIR lowering.
func oracle(t *testing.T, dir string) func([]float32) []float64 {
	b, err := os.ReadFile(dir + "/heir/input.mlir")
	if err != nil {
		t.Fatal(err)
	}
	mm := regexp.MustCompile(`(?s)%w[01] = arith\.constant dense<(\[.*?\])> : tensor`).FindAllSubmatch(b, -1)
	if len(mm) != 2 {
		t.Fatalf("found %d public matrices", len(mm))
	}
	var w [2][][]float64
	for i, m := range mm {
		if err = json.Unmarshal(m[1], &w[i]); err != nil {
			t.Fatal(err)
		}
	}
	return func(x []float32) []float64 {
		in := make([]float64, len(x))
		for i, v := range x {
			in[i] = float64(v)
		}
		for l, m := range w {
			y := make([]float64, len(m))
			for i, row := range m {
				for j, c := range row {
					y[i] += c * in[j]
				}
				if l == 0 {
					y[i] /= 1 + math.Exp(-y[i])
				}
			}
			in = y
		}
		return in
	}
}

type metric struct {
	Linf, MSE, VsHeirLinf, MeanEvalMs      float64
	InputLevel, OutputLevel                int
	OutputLogScale                         float64
	Galois                                 []uint64
	SerializedGaloisBytes, Trials, Outputs int
}

func experiment(t *testing.T, work string, n int, arms [3]arm) {
	t.Helper()
	dir := "../../build/pass1/" + work
	reference := oracle(t, dir)
	data, err := os.ReadFile(dir + "/plan.json")
	if err != nil {
		t.Fatal(err)
	}
	var p plan
	if err = json.Unmarshal(data, &p); err != nil {
		t.Fatal(err)
	}
	if p.Bootstrap.Status != "not_applicable" || len(p.Regions) != 2 || p.Regions[0].Level != 7 || p.Regions[1].Level != 2 {
		t.Fatal("unexpected graph budget or bootstrap inventory")
	}
	for i, a := range []planArm{p.Independent, p.Joint} {
		var keys []uint64
		for _, k := range a.Keys {
			keys = append(keys, k.Galois)
		}
		if !slices.Equal(keys, arms[i+1].keys) || a.KeyBytes != arms[i+1].keyBytes {
			t.Fatal("generated keys differ from scoped plan", keys, arms[i+1].keys, a.KeyBytes, arms[i+1].keyBytes)
		}
	}
	if work == "strided64" {
		if !slices.Equal(p.Regions[0].Diagonals, []int{0, 1, 2, 5}) || !slices.Equal(p.Regions[1].Diagonals, []int{0, 4}) ||
			p.Independent.Routes[0].S == p.Joint.Routes[0].S || len(arms[1].keys) != 4 || len(arms[2].keys) != 3 || arms[2].keyBytes >= arms[1].keyBytes {
			t.Fatal("missing first-layer change and physical key saving")
		}
		shared := false
		for _, k := range p.Joint.Keys {
			if k.Galois == 625 && k.Refs == 2 {
				shared = true
			}
		}
		if !shared {
			t.Fatal("rotation 4 must be shared across both transforms")
		}
	}
	if work == "reuse64" {
		if !slices.Equal(p.Regions[0].Diagonals, []int{0, 7, 9}) || !slices.Equal(p.Regions[1].Diagonals, []int{0, 6, 8}) ||
			p.Independent.Routes[0].S == p.Joint.Routes[0].S || len(arms[0].keys) != 5 || len(arms[1].keys) != 4 || len(arms[2].keys) != 3 {
			t.Fatal("unexpected physical inventory on the double-reuse workload")
		}
		shared := 0
		for _, k := range p.Joint.Keys {
			if k.Refs == 2 {
				shared++
			}
		}
		if shared != 2 {
			t.Fatal("expected two shared rotations", p.Joint.Keys)
		}
	}
	rng := rand.New(rand.NewPCG(314159, 271828))
	names := [3]string{"heir", "independent", "joint"}
	var stats [3]metric
	for trial := 0; trial < 64; trial++ {
		x := make([]float32, n)
		for j := range x {
			switch {
			case n == 4 && trial < 16:
				x[j] = float32(2*((trial>>j)&1) - 1)
			case n == 64 && trial < 2:
				x[j] = float32(2*trial - 1)
			case n == 64 && trial < 4:
				x[j] = float32(2*((j/(trial-1))%2) - 1)
			case n == 64 && trial < 16:
				if j == (5*trial)%n {
					x[j] = 1
				}
			default:
				x[j] = float32(2*rng.Float64() - 1)
			}
		}
		plain := reference(x)
		var stock []float32
		var stockScale float64
		// Interleave arms; key generation, preprocessing, encrypt/decrypt are not timed.
		for i, a := range arms {
			got, in, out, scale, dt := a.run(x)
			if len(got) != n || in != 7 || out != 0 {
				t.Fatalf("%s: output/levels %d %d->%d", names[i], len(got), in, out)
			}
			if i == 0 {
				stock = got
				stockScale = scale
			}
			if math.IsNaN(scale) || math.IsInf(scale, 0) || math.Abs(scale-stockScale) > 1e-8 {
				t.Fatalf("%s: scale drift", names[i])
			}
			s := &stats[i]
			if s.Trials > 0 && s.OutputLogScale != scale {
				t.Fatalf("%s: unstable output scale", names[i])
			}
			s.InputLevel, s.OutputLevel, s.OutputLogScale = in, out, scale
			s.MeanEvalMs += float64(dt) / float64(time.Millisecond)
			s.Trials++
			s.Outputs += n
			for j, v := range got {
				e := math.Abs(float64(v) - plain[j])
				diff := math.Abs(float64(v) - float64(stock[j]))
				if math.IsNaN(e) || e > tolerance || diff > tolerance {
					t.Fatalf("%s trial %d slot %d: got %g, oracle %g, HEIR %g", names[i], trial, j, v, plain[j], stock[j])
				}
				s.Linf = math.Max(s.Linf, e)
				s.MSE += e * e
				s.VsHeirLinf = math.Max(s.VsHeirLinf, diff)
			}
		}
	}
	for i, s := range stats {
		s.MSE /= float64(s.Outputs)
		s.MeanEvalMs /= float64(s.Trials)
		s.Galois = arms[i].keys
		s.SerializedGaloisBytes = arms[i].keyBytes
		row, err := json.Marshal(map[string]any{"workload": work, "arm": names[i], "metrics": s})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("SNABF_PASS1 %s", row)
	}
}

func TestControl(t *testing.T) {
	experiment(t, "control", 4, [3]arm{
		newArm(t, hc.Linear_silu_linear__configure, hc.Linear_silu_linear__preprocessing, hc.Linear_silu_linear__preprocessed, hc.Linear_silu_linear__encrypt__arg0, hc.Linear_silu_linear__decrypt__result0),
		newArm(t, ic.Linear_silu_linear__configure, ic.Linear_silu_linear__preprocessing, ic.Linear_silu_linear__preprocessed, ic.Linear_silu_linear__encrypt__arg0, ic.Linear_silu_linear__decrypt__result0),
		newArm(t, jc.Linear_silu_linear__configure, jc.Linear_silu_linear__preprocessing, jc.Linear_silu_linear__preprocessed, jc.Linear_silu_linear__encrypt__arg0, jc.Linear_silu_linear__decrypt__result0)})
}
func TestStrided64(t *testing.T) {
	experiment(t, "strided64", 64, [3]arm{
		newArm(t, hs.Linear_silu_linear__configure, hs.Linear_silu_linear__preprocessing, hs.Linear_silu_linear__preprocessed, hs.Linear_silu_linear__encrypt__arg0, hs.Linear_silu_linear__decrypt__result0),
		newArm(t, is.Linear_silu_linear__configure, is.Linear_silu_linear__preprocessing, is.Linear_silu_linear__preprocessed, is.Linear_silu_linear__encrypt__arg0, is.Linear_silu_linear__decrypt__result0),
		newArm(t, js.Linear_silu_linear__configure, js.Linear_silu_linear__preprocessing, js.Linear_silu_linear__preprocessed, js.Linear_silu_linear__encrypt__arg0, js.Linear_silu_linear__decrypt__result0)})
}
func TestReuse64(t *testing.T) {
	experiment(t, "reuse64", 64, [3]arm{
		newArm(t, hr.Linear_silu_linear__configure, hr.Linear_silu_linear__preprocessing, hr.Linear_silu_linear__preprocessed, hr.Linear_silu_linear__encrypt__arg0, hr.Linear_silu_linear__decrypt__result0),
		newArm(t, ir.Linear_silu_linear__configure, ir.Linear_silu_linear__preprocessing, ir.Linear_silu_linear__preprocessed, ir.Linear_silu_linear__encrypt__arg0, ir.Linear_silu_linear__decrypt__result0),
		newArm(t, jr.Linear_silu_linear__configure, jr.Linear_silu_linear__preprocessing, jr.Linear_silu_linear__preprocessed, jr.Linear_silu_linear__encrypt__arg0, jr.Linear_silu_linear__decrypt__result0)})
}
