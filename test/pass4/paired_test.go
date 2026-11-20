//go:build snabf_generated

package comparison

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/gadget"
)

// The generated sibling adapter binds newDirect, legacyConfigure, legacyPrepare
// and legacyEval by Go package, not by rewriting either generated evaluator.
func TestPairedCiphertexts(t *testing.T) {
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	r, err := newDirect()
	check(err)
	eval, p, encoder := r.SourceEvaluation()
	if r.Program.Source != r.Program.Target || eval != r.Evaluator(r.Program.Source) {
		t.Fatal("historical comparator only supports one ring")
	}
	ref, rp, _, _, _ := legacyConfigure()
	if p.N() != rp.N() || p.LogDefaultScale() != rp.LogDefaultScale() || !slices.Equal(p.Q(), rp.Q()) || !slices.Equal(p.P(), rp.P()) {
		t.Fatal("physical parameter mismatch")
	}
	sharedKeys := sha256.New()
	inventory := func(e *ckks.Evaluator) map[uint64]int {
		ids := e.GetGaloisKeysList()
		slices.Sort(ids)
		result := make(map[uint64]int, len(ids))
		for _, id := range ids {
			k, err := e.GetGaloisKey(id)
			check(err)
			b, err := k.MarshalBinary()
			check(err)
			result[id] = len(b)
			if e == eval {
				sharedKeys.Write(b)
			}
		}
		return result
	}
	a, b := inventory(eval), inventory(ref)
	relin, err := eval.GetRelinearizationKey()
	check(err)
	rawRelin, err := relin.MarshalBinary()
	check(err)
	sharedKeys.Write(rawRelin)
	sharedID := hex.EncodeToString(sharedKeys.Sum(nil))
	if len(a) != len(b) {
		t.Fatal("different Galois key counts", a, b)
	}
	for id, n := range a {
		if b[id] != n {
			t.Fatal("different physical Galois key/size", id)
		}
	}
	lt, pt := legacyPrepare(p, encoder)
	var corpus []struct{ X, Truth []float64 }
	raw, err := os.ReadFile(os.Getenv("SNABF_CORPUS"))
	check(err)
	check(json.Unmarshal(raw, &corpus))
	if len(corpus) != 64 {
		t.Fatal("incomplete corpus")
	}
	file, err := os.Create(os.Getenv("SNABF_EVIDENCE") + "/trials.jsonl")
	check(err)
	defer file.Close()
	buffer := bufio.NewWriter(file)
	rows := json.NewEncoder(buffer)
	var absolute, sq, online [2]float64
	var paired, sumDiff float64
	var checked int
	var observed [2][]gadget.Event
	type physical struct {
		N, Level int
		LogScale float64
	}
	var final physical
	ctID := func(x *rlwe.Ciphertext) string {
		b, err := x.MarshalBinary()
		check(err)
		digest := sha256.Sum256(b)
		return hex.EncodeToString(digest[:])
	}
	run := func(arm int, x *rlwe.Ciphertext) *rlwe.Ciphertext {
		if arm == 0 {
			y, err := r.Eval(x)
			check(err)
			return y
		}
		z := legacyEval(eval, p, encoder, []*rlwe.Ciphertext{x}, lt, pt)
		if len(z) != 1 {
			t.Fatal("wrong legacy result count")
		}
		return z[0]
	}
	var audit struct {
		Sink         int
		Fingerprint  string `json:"emit_sha256"`
		Instructions []struct {
			ID       string
			Level, K int
			Scale    float64 `json:"physical_log_scale"`
		}
	}
	raw, err = os.ReadFile(os.Getenv("SNABF_PROGRAM"))
	check(err)
	check(json.Unmarshal(raw, &audit))
	if audit.Fingerprint != r.Program.Fingerprint || audit.Sink < 0 || audit.Sink >= len(audit.Instructions) {
		t.Fatal("executable/audit mismatch")
	}
	want := audit.Instructions[audit.Sink]
	for trial, c := range corpus {
		x, err := r.Encrypt(c.X)
		check(err)
		var decoded [2][]float64
		var ns [2]int64
		var output [2]physical
		var order [2]int
		var copies [2]string
		inputID := ctID(x)
		for off := 0; off < 2; off++ {
			arm := (trial + off) % 2
			order[off] = arm
			copy := x.CopyNew()
			copies[arm] = ctID(copy)
			if copies[arm] != inputID {
				t.Fatal("arms did not receive a copy of the same ciphertext")
			}
			start := time.Now()
			y := run(arm, copy)
			ns[arm] = time.Since(start).Nanoseconds()
			online[arm] += float64(ns[arm]) / 1e6
			final = physical{y.Value[0].N(), y.Level(), y.LogScale()}
			output[arm] = final
			if final.N != p.N() || final.Level != want.Level || math.IsNaN(final.LogScale) || math.Abs(final.LogScale-want.Scale) > 1e-7 || y.Slots() != r.Program.Slots {
				t.Fatal("result ABI", arm, final)
			}
			decoded[arm], err = r.Decode(y)
			check(err)
			if len(decoded[arm]) != r.Program.Slots || len(c.Truth) != r.Program.Width {
				t.Fatal("invalid slots")
			}
		}
		for j := range decoded[0] {
			d := math.Abs(decoded[0][j] - decoded[1][j])
			paired = math.Max(paired, d)
			sumDiff += d * d
			for arm := 0; arm < 2; arm++ {
				v := math.Abs(decoded[arm][j] - c.Truth[j%len(c.Truth)])
				if math.IsNaN(v) || math.IsInf(v, 0) || v > 1./4096 {
					t.Fatalf("trial=%d slot=%d arm=%d error=%g", trial, j, arm, v)
				}
				absolute[arm] = math.Max(absolute[arm], v)
				sq[arm] += v * v
			}
			checked++
		}
		check(rows.Encode(map[string]any{"trial": trial, "decoded_direct": decoded[0], "decoded_legacy": decoded[1], "online_ns": ns,
			"output_direct_legacy": output, "eval_order": order, "input_sha256": inputID, "copy_sha256_direct_legacy": copies, "shared_eval_keys_sha256": sharedID}))
		if trial == 0 {
			for arm := 0; arm < 2; arm++ {
				func() {
					gadget.Observe = func(ev gadget.Event) { observed[arm] = append(observed[arm], ev) }
					defer func() { gadget.Observe = nil }()
					run(arm, x.CopyNew())
				}()
			}
		}
	}
	check(buffer.Flush())
	for arm := 0; arm < 2; arm++ {
		expected := map[string]int{}
		for _, e := range audit.Instructions {
			if e.K > 0 {
				expected[e.ID] = e.K
			}
		}
		for _, ev := range observed[arm] {
			k, ok := expected[ev.ID]
			if !ok || ev.RowsSkipped != k || ev.ResidualMax <= 0 {
				t.Fatal("unplanned gadget execution", arm, ev)
			}
			delete(expected, ev.ID)
		}
		if len(expected) != 0 {
			t.Fatal("missing gadget executions", arm, expected)
		}
		sq[arm] /= float64(checked)
		online[arm] /= float64(len(corpus))
	}
	if paired > 1./4096 {
		t.Fatal("paired outputs differ above tolerance", paired)
	}
	raw, err = json.MarshalIndent(map[string]any{"trials": len(corpus), "slots_per_arm": checked, "linf_direct_legacy": absolute, "mse_direct_legacy": sq, "online_ms_direct_legacy": online, "paired_linf": paired, "paired_mse": sumDiff / float64(checked), "galois_bytes_by_id": a, "gadget_events_direct_legacy": observed, "actual_output": final, "same_ciphertext_and_keys": true,
		"keys": r.Keys, "shared_eval_keys_sha256": sharedID, "untimed_probe_evals": 2}, "", "  ")
	check(err)
	check(os.WriteFile(os.Getenv("SNABF_EVIDENCE")+"/metrics.json", raw, 0644))
	t.Logf("%d paired inferences, %d slots per arm, max vs truth %g/%g, max paired %g", len(corpus), checked, absolute[0], absolute[1], paired)
}
