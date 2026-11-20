package unified_test

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"slices"
	"testing"
	"time"

	cap2 "snabf.local/compiler/build/unified/reuse64/cap2"
	"snabf.local/compiler/build/unified/reuse64/heir"
)

// The sole input change to the compiler is a resource contract on the sink.
func TestSinkFeedbackExecution(t *testing.T) {
	e, p, c, enc, dec := cap2.Linear_silu_linear__configure()
	h, hp, hc, he, hd := baseline.Linear_silu_linear__configure()
	lt, pt := cap2.Linear_silu_linear__preprocessing(p, c)
	hlt, hpt := baseline.Linear_silu_linear__preprocessing(hp, hc)
	var audit struct {
		Rounds   int
		Keys     []struct{ Galois uint64 }
		KeyBytes int `json:"key_bytes"`
	}
	b, err := os.ReadFile("../../build/unified/reuse64/cap2/audit.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &audit); err != nil {
		t.Fatal(err)
	}
	keys := e.GetGaloisKeysList()
	slices.Sort(keys)
	var want []uint64
	size := 0
	for _, k := range audit.Keys {
		want = append(want, k.Galois)
		g, err := e.GetGaloisKey(k.Galois)
		if err != nil {
			t.Fatal(err)
		}
		data, err := g.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		size += len(data)
	}
	if len(keys) != 4 || !slices.Equal(keys, want) || audit.Rounds != 2 || size != audit.KeyBytes {
		t.Fatalf("wrong executed feedback plan: %v, %d B", keys, size)
	}
	var ws [2][][]float64
	source, err := os.ReadFile("../../build/unified/reuse64/heir/input.mlir")
	if err != nil {
		t.Fatal(err)
	}
	rows := regexp.MustCompile(`(?s)%w[01] = arith\.constant dense<(\[.*?\])> : tensor`).FindAllSubmatch(source, -1)
	if len(rows) != 2 {
		t.Fatal("missing source weights")
	}
	for i, w := range rows {
		if err := json.Unmarshal(w[1], &ws[i]); err != nil {
			t.Fatal(err)
		}
	}
	var maxDiff, linf, sse, ms, logScale float64
	for trial := 0; trial < 4; trial++ {
		x := make([]float32, 64)
		for i := range x {
			x[i] = float32(math.Sin(float64((i + 1) * (trial + 1))))
		}
		cx := cap2.Linear_silu_linear__encrypt__arg0(e, p, c, enc, x)
		start := time.Now()
		cy := cap2.Linear_silu_linear__preprocessed(e, p, c, cx, lt, pt)
		ms += float64(time.Since(start)) / float64(time.Millisecond)
		hx := baseline.Linear_silu_linear__encrypt__arg0(h, hp, hc, he, x)
		hy := baseline.Linear_silu_linear__preprocessed(h, hp, hc, hx, hlt, hpt)
		y := cap2.Linear_silu_linear__decrypt__result0(e, p, c, dec, cy)
		ref := baseline.Linear_silu_linear__decrypt__result0(h, hp, hc, hd, hy)
		logScale = cy[0].LogScale()
		if len(y) != 64 || len(ref) != 64 || cx[0].Level() != 7 || cy[0].Level() != 0 || math.IsNaN(logScale) || math.IsInf(logScale, 0) || math.Abs(logScale-hy[0].LogScale()) > 1e-8 {
			t.Fatal("invalid output metadata")
		}
		plain := make([]float64, len(x))
		for i, v := range x {
			plain[i] = float64(v)
		}
		for l, w := range ws {
			out := make([]float64, len(w))
			for i, row := range w {
				for j, c := range row {
					out[i] += c * plain[j]
				}
				if l == 0 {
					out[i] /= 1 + math.Exp(-out[i])
				}
			}
			plain = out
		}
		for i, v := range y {
			abs := math.Abs(float64(v) - plain[i])
			if math.IsNaN(abs) || abs > math.Ldexp(1, -12) {
				t.Fatalf("cap2 truth error slot %d: %g", i, abs)
			}
			linf = math.Max(linf, abs)
			sse += abs * abs
			d := math.Abs(float64(v) - float64(ref[i]))
			if math.IsNaN(d) || d > math.Ldexp(1, -12) {
				t.Fatalf("slot %d: %g", i, d)
			}
			maxDiff = math.Max(maxDiff, d)
		}
	}
	row, err := json.Marshal(map[string]any{"workload": "reuse64", "arm": "cap2", "metrics": map[string]any{"Trials": 4, "Outputs": 256, "Linf": linf, "MSE": sse / 256, "VsHeirLinf": maxDiff, "MeanEvalMs": ms / 4, "Galois": keys, "SerializedGaloisBytes": size, "InputLevel": 7, "OutputLevel": 0, "OutputLogScale": logScale}})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SNABF_PASS1 %s", row)
}
