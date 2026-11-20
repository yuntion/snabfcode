package pass3_test

import (
	"encoding/json"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math"
	"os"
	"path/filepath"
	"slices"
	"snabf.local/compiler/src/backend/gadget"
	"snabf.local/compiler/src/backend/tower"
	"testing"
	"time"
)

type program struct {
	name string
	cfg  func() (*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor)
	prep func(ckks.Parameters, *ckks.Encoder) ([]tower.LinearTransformation, []*rlwe.Plaintext)
	eval func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, []*rlwe.Ciphertext, []tower.LinearTransformation, []*rlwe.Plaintext) []*rlwe.Ciphertext
	enc  func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, []float32) []*rlwe.Ciphertext
	dec  func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Decryptor, []*rlwe.Ciphertext) []float32
}
type node struct {
	ID       string
	Level, K int
	Scale    float64 `json:"physical_log_scale"`
}
type plan struct {
	Ring, Scale, Sink int
	Q, P              []uint64
	Edges             []node
}
type state struct {
	program
	p       plan
	e       *ckks.Evaluator
	par     ckks.Parameters
	encoder *ckks.Encoder
	encKey  *rlwe.Encryptor
	decKey  *rlwe.Decryptor
	lt      []tower.LinearTransformation
	pt      []*rlwe.Plaintext
}
type stats struct {
	Slots, Misses, Trials                       int
	Linf, MSE, OnlineMS, ObservedOutputLogScale float64
	OutputLevel                                 int
	Events                                      []gadget.Event
	PairedLinf, PairedMeanSquare                float64
}

func read(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}
func experiment(t *testing.T, dir string, arms []program) {
	var cases []struct {
		X     []float32
		Truth []float64
	}
	read(t, filepath.Join(dir, "corpus.json"), &cases)
	if len(cases) != 64 {
		t.Fatal("missing frozen trials")
	}
	var all []state
	st := make([]stats, len(arms))
	var rows []*json.Encoder
	for _, pr := range arms {
		s := state{program: pr}
		read(t, filepath.Join(dir, pr.name, "plan.json"), &s.p)
		if pr.name == "matched_exact" {
			j := all[1]
			s.e, s.par, s.encoder, s.encKey, s.decKey = j.e, j.par, j.encoder, j.encKey, j.decKey
		} else {
			s.e, s.par, s.encoder, s.encKey, s.decKey = pr.cfg()
		}
		if s.p.Ring != s.par.N() || s.p.Scale != s.par.LogDefaultScale() || !slices.Equal(s.p.Q, s.par.Q()) || !slices.Equal(s.p.P, s.par.P()) {
			t.Fatal("stale actual parameters", pr.name)
		}
		s.lt, s.pt = pr.prep(s.par, s.encoder)
		all = append(all, s)
		f, err := os.Create(filepath.Join(dir, pr.name, "trials.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { f.Close() })
		rows = append(rows, json.NewEncoder(f))
	}
	for i, c := range cases {
		var cts [3][]*rlwe.Ciphertext
		for j := 0; j < 2; j++ {
			s := &all[j]
			cts[j] = s.enc(s.e, s.par, s.encoder, s.encKey, c.X)
		}
		cts[2] = cts[1]
		var full [3][]float64
		for off := range all {
			j := (off + i) % len(all)
			s := &all[j]
			z := &st[j]
			input := []*rlwe.Ciphertext{cts[j][0].CopyNew()}
			if input[0].Level() != len(s.p.Q)-1 || math.Abs(input[0].LogScale()-float64(s.p.Scale)) > 1e-9 {
				t.Fatal("input ABI")
			}
			start := time.Now()
			cy := s.eval(s.e, s.par, s.encoder, input, s.lt, s.pt)
			ns := time.Since(start).Nanoseconds()
			z.OnlineMS += float64(ns) / 1e6
			want := s.p.Edges[s.p.Sink]
			if len(cy) != 1 || cy[0].Level() != want.Level || math.Abs(cy[0].LogScale()-want.Scale) > 1e-7 {
				t.Fatalf("output ABI %s L%d/%g want L%d/%g", s.name, cy[0].Level(), cy[0].LogScale(), want.Level, want.Scale)
			}
			z.OutputLevel = cy[0].Level()
			z.ObservedOutputLogScale = cy[0].LogScale()
			v := s.dec(s.e, s.par, s.encoder, s.decKey, cy)
			if len(v) != len(c.Truth) {
				t.Fatal("wrong output width")
			}
			decoded := make([]float64, len(v))
			for h, x := range v {
				decoded[h] = float64(x)
				err := math.Abs(decoded[h] - c.Truth[h])
				if math.IsNaN(err) || math.IsInf(err, 0) {
					t.Fatal("nonfinite output")
				}
				if err > 1./4096 {
					z.Misses++
				}
				z.Linf = math.Max(z.Linf, err)
				z.MSE += err * err
				z.Slots++
			}
			z.Trials++
			// Compare the same actual ciphertext before HEIR narrows its public ABI to float32.
			full[j] = make([]float64, s.par.MaxSlots())
			if err := s.encoder.Decode(s.decKey.DecryptNew(cy[0]), full[j]); err != nil {
				t.Fatal(err)
			}
			full[j] = full[j][:len(v)]
			if err := rows[j].Encode(map[string]any{"trial": i, "decoded": decoded, "decoded64": full[j], "online_ns": ns, "level": cy[0].Level(), "log_scale": cy[0].LogScale()}); err != nil {
				t.Fatal(err)
			}
			if i == 0 {
				func() {
					gadget.Observe = func(ev gadget.Event) { z.Events = append(z.Events, ev) }
					defer func() { gadget.Observe = nil }()
					s.eval(s.e, s.par, s.encoder, []*rlwe.Ciphertext{cts[j][0].CopyNew()}, s.lt, s.pt)
				}()
				expected := map[string]node{}
				for _, r := range s.p.Edges {
					if r.K > 0 {
						expected[r.ID] = r
					}
				}
				for _, ev := range z.Events {
					r, ok := expected[ev.ID]
					if !ok || ev.RowsSkipped != r.K || ev.RowsExecuted != r.Level+1-r.K || ev.ResidualMax <= 0 || math.Abs(ev.LogScale-r.Scale) > 1e-7 {
						t.Fatalf("bad actual site: %+v plan %+v", ev, r)
					}
					delete(expected, ev.ID)
				}
				if len(expected) != 0 {
					t.Fatalf("unexecuted truncation: %v", expected)
				}
			}
		}
		for h, x := range full[1] {
			diff := math.Abs(x - full[2][h])
			st[1].PairedLinf = math.Max(st[1].PairedLinf, diff)
			st[1].PairedMeanSquare += diff * diff
		}
	}
	for j, s := range all {
		z := &st[j]
		z.MSE /= float64(z.Slots)
		z.OnlineMS /= float64(z.Trials)
		z.PairedMeanSquare /= float64(z.Slots)
		b, _ := json.MarshalIndent(z, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, s.name, "metrics.json"), b, 0644); err != nil {
			t.Fatal(err)
		}
		t.Log(fmt.Sprintf("%s/%s: %d/%d misses Linf=%g MSE=%g mean=%.2fms", dir, s.name, z.Misses, z.Slots, z.Linf, z.MSE, z.OnlineMS))
		if z.Misses != 0 {
			t.Fail()
		}
		if s.name == "joint" && (len(z.Events) == 0 || z.PairedLinf == 0) {
			t.Fatal("missing paired approximation evidence")
		}
	}
}
