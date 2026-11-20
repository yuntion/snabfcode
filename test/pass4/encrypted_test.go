//go:build snabf_generated

package planned

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"testing"
	"time"
)

// All encrypted operators run before Decode; the oracle is used only here.
func TestEncrypted(t *testing.T) {
	var cases []struct{ X, Truth []float64 }
	data, err := os.ReadFile(os.Getenv("SNABF_CORPUS"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) != 64 {
		t.Fatal("incomplete frozen corpus")
	}
	r, err := New()
	if err != nil {
		t.Fatal(err)
	}
	dir := os.Getenv("SNABF_EVIDENCE")
	file, err := os.Create(dir + "/trials.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	w := bufio.NewWriter(file)
	enc := json.NewEncoder(w)
	var peak, sum, ms float64
	var slots, logical, miss int
	var scale float64
	var ring, level int
	for j, c := range cases {
		if len(c.Truth) != r.Program.Width {
			t.Fatal("oracle width")
		}
		x, err := r.Encrypt(c.X)
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		y, err := r.Eval(x)
		ns := time.Since(start).Nanoseconds()
		if err != nil {
			t.Fatal(err)
		}
		ring = y.Value[0].N()
		level = y.Level()
		scale = y.LogScale()
		vals, err := r.Decode(y)
		if err != nil {
			t.Fatal(err)
		}
		for k, v := range vals {
			d := math.Abs(v - c.Truth[k%len(c.Truth)])
			if math.IsNaN(d) || math.IsInf(d, 0) {
				t.Fatal("nonfinite output")
			}
			if d > 1./4096 {
				miss++
			}
			sum += d * d
			peak = math.Max(peak, d)
		}
		slots += len(vals)
		logical += len(c.Truth)
		ms += float64(ns) / 1e6
		if err = enc.Encode(map[string]any{"trial": j, "decoded64": vals, "truth_one_period": c.Truth, "online_ns": ns, "ring": ring, "level": level, "log_scale": scale}); err != nil {
			t.Fatal(err)
		}
	}
	if err = w.Flush(); err != nil {
		t.Fatal(err)
	}
	metrics := map[string]any{"trials": len(cases), "physical_encoded_slots_checked": slots, "logical_outputs": logical, "misses": miss, "linf": peak, "mse": sum / float64(slots), "online_ms": ms / float64(len(cases)), "actual_output_ring": ring, "actual_level": level, "actual_log_scale": scale, "keys": r.Keys}
	b, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(dir+"/metrics.json", b, 0644); err != nil {
		t.Fatal(err)
	}
	if miss != 0 {
		t.Fatalf("%d threshold violations, peak %g", miss, peak)
	}
	t.Logf("N=%d, %d logical outputs, %d encoded slots, max=%g, mean=%.2f ms", ring, logical, slots, peak, ms/float64(len(cases)))
}
