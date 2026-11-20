package pass2_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const tolerance = 1.0 / (1 << 12)

type configure func() (*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor)
type encrypt func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Encryptor, []float32) []*rlwe.Ciphertext
type decrypt func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, *rlwe.Decryptor, []*rlwe.Ciphertext) []float32

type observed struct {
	Trial          int       `json:"trial"`
	InputLevel     int       `json:"input_level"`
	OutputLevel    int       `json:"output_level"`
	InputLogScale  float64   `json:"input_log_scale"`
	OutputLogScale float64   `json:"output_log_scale"`
	EncryptNS      int64     `json:"encrypt_ns"`
	EvalNS         int64     `json:"eval_ns"`
	DecryptNS      int64     `json:"decrypt_ns"`
	Decrypted      []float32 `json:"-"`
}

type arm struct {
	name string
	run  func([]float32) observed
}

type cipherABI struct {
	InputLevel     int     `json:"input_level"`
	OutputLevel    int     `json:"output_level"`
	InputLogScale  float64 `json:"input_log_scale"`
	OutputLogScale float64 `json:"output_log_scale"`
}

type miss struct {
	Trial     int     `json:"trial"`
	Slot      int     `json:"slot"`
	Decrypted float64 `json:"decrypted"`
	Truth     float64 `json:"truth"`
	AbsError  float64 `json:"abs_error"`
}

type accuracy struct {
	Tolerance           float64 `json:"tolerance"`
	LogicalSlotsChecked int     `json:"logical_slots_checked"`
	SlotsOverTolerance  int     `json:"slots_over_tolerance"`
	Linf                float64 `json:"linf"`
	SumSquaredError     float64 `json:"sum_squared_error"`
	MSE                 float64 `json:"mse"`
	PrecisionPassed     bool    `json:"precision_passed"`
	HardPrecisionGate   bool    `json:"hard_precision_gate"`
	Status              string  `json:"accuracy_status"`
	FirstFailure        *miss   `json:"first_failure"`
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func hardPrecisionGate(work, name string) bool { return work != "residual" || name != "heir" }

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func readJSON(t *testing.T, path string, dest any) {
	t.Helper()
	b, err := os.ReadFile(path)
	check(t, err)
	if err := json.Unmarshal(b, dest); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func checkEdge(t *testing.T, edge string, level int, logScale float64, wantLevel int, wantLogScale float64) {
	t.Helper()
	if level != wantLevel || !finite(logScale) || math.Abs(logScale-wantLogScale) > 0.25 {
		t.Fatalf("%s ABI: observed L%d/logscale %.9g, want L%d/%.9g (+/-0.25 scale bits)", edge, level, logScale, wantLevel, wantLogScale)
	}
}

func digest(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	check(t, err)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func newArm[T any](t *testing.T, dir, name string, cfg configure,
	prepare func(ckks.Parameters, *ckks.Encoder) ([]T, []*rlwe.Plaintext),
	eval func(*ckks.Evaluator, ckks.Parameters, *ckks.Encoder, []*rlwe.Ciphertext, []T, []*rlwe.Plaintext) []*rlwe.Ciphertext,
	enc encrypt, dec decrypt) arm {
	t.Helper()
	start := time.Now()
	e, p, encoder, encryptor, decryptor := cfg()
	cfgNS := time.Since(start).Nanoseconds()
	var profile struct {
		LogN            int
		Q, P            []uint64
		LogDefaultScale int
	}
	readJSON(t, filepath.Join(dir, "profile.json"), &profile)
	if p.LogN() != profile.LogN || !slices.Equal(p.Q(), profile.Q) || !slices.Equal(p.P(), profile.P) || p.LogDefaultScale() != profile.LogDefaultScale {
		t.Fatalf("%s: real ring/Q/P/default scale disagree with the declared profile", name)
	}
	abi := cipherABI{-1, -1, math.NaN(), math.NaN()}
	readJSON(t, filepath.Join(dir, name, "abi.json"), &abi)
	if abi.OutputLevel < 0 || abi.InputLevel < abi.OutputLevel || abi.InputLevel > p.MaxLevel() || !finite(abi.InputLogScale) || !finite(abi.OutputLogScale) {
		t.Fatalf("%s: invalid declared ciphertext ABI: %+v", name, abi)
	}
	start = time.Now()
	lt, pts := prepare(p, encoder)
	prepNS := time.Since(start).Nanoseconds()
	if len(lt) != 2 {
		t.Fatalf("%s: expected two physical linear transforms, got %d", name, len(lt))
	}
	ids := e.GetGaloisKeysList()
	slices.Sort(ids)
	keys := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		k, err := e.GetGaloisKey(id)
		check(t, err)
		b, err := k.MarshalBinary()
		check(t, err)
		keys = append(keys, map[string]any{"galois": id, "serialized_bytes": len(b)})
	}
	rlk, err := e.GetRelinearizationKey()
	check(t, err)
	rb, err := rlk.MarshalBinary()
	check(t, err)
	meta := map[string]any{"corpus_sha256": digest(t, filepath.Join(dir, "corpus.json")),
		"program_sha256": digest(t, filepath.Join(dir, name, "model.go")), "galois_keys": keys,
		"relinearization_key_serialized_bytes": len(rb), "logN": p.LogN(), "Q": p.Q(), "P": p.P(),
		"default_log_scale": p.LogDefaultScale(), "configure_ns": cfgNS, "preprocessing_ns": prepNS}
	b, err := json.MarshalIndent(meta, "", "  ")
	check(t, err)
	check(t, os.WriteFile(filepath.Join(dir, name, "runtime.json"), b, 0644))
	return arm{name, func(x []float32) observed {
		var row observed
		start := time.Now()
		ct := enc(e, p, encoder, encryptor, x)
		row.EncryptNS = time.Since(start).Nanoseconds()
		if len(ct) != 1 || ct[0] == nil {
			t.Fatalf("%s: expected one packed input", name)
		}
		row.InputLevel, row.InputLogScale = ct[0].Level(), ct[0].LogScale()
		checkEdge(t, name+" input", row.InputLevel, row.InputLogScale, abi.InputLevel, abi.InputLogScale)
		start = time.Now()
		cy := eval(e, p, encoder, ct, lt, pts)
		row.EvalNS = time.Since(start).Nanoseconds()
		if len(cy) != 1 || cy[0] == nil {
			t.Fatalf("%s: expected one packed result", name)
		}
		row.OutputLevel, row.OutputLogScale = cy[0].Level(), cy[0].LogScale()
		checkEdge(t, name+" output", row.OutputLevel, row.OutputLogScale, abi.OutputLevel, abi.OutputLogScale)
		start = time.Now()
		row.Decrypted = dec(e, p, encoder, decryptor, cy)
		row.DecryptNS = time.Since(start).Nanoseconds()
		return row
	}}
}

func experiment(t *testing.T, dir string, width int, arms []arm) {
	t.Helper()
	var cases []struct {
		Trial int
		X     []float32
		Truth []float64
	}
	readJSON(t, filepath.Join(dir, "corpus.json"), &cases)
	if len(cases) != 64 || width <= 0 || len(arms) == 0 {
		t.Fatal("expected 64 inputs and at least one executable")
	}
	writers := make([]*json.Encoder, len(arms))
	work := filepath.Base(filepath.Clean(dir))
	stats := make([]accuracy, len(arms))
	for j, a := range arms {
		stats[j] = accuracy{Tolerance: tolerance, HardPrecisionGate: hardPrecisionGate(work, a.name)}
		file, err := os.Create(filepath.Join(dir, a.name, "trials.jsonl"))
		check(t, err)
		t.Cleanup(func() { check(t, file.Close()) })
		writers[j] = json.NewEncoder(file)
	}
	for i, c := range cases {
		if i != c.Trial || len(c.Truth) != width || len(c.X) != width {
			t.Fatalf("bad corpus trial %d", i)
		}
		for slot := range c.Truth {
			if !finite(c.Truth[slot]) || !finite(float64(c.X[slot])) {
				t.Fatalf("nonfinite corpus trial %d slot %d", i, slot)
			}
		}
		// Rotate start order; each arm sees the same input with fresh encryption.
		for k := range arms {
			j := (i + k) % len(arms)
			r := arms[j].run(c.X)
			r.Trial = i
			if len(r.Decrypted) != len(c.Truth) {
				t.Fatalf("%s trial %d: wrong logical width", arms[j].name, i)
			}
			if r.EncryptNS <= 0 || r.EvalNS <= 0 || r.DecryptNS <= 0 {
				t.Fatalf("%s trial %d: invalid timing", arms[j].name, i)
			}
			// Encode exact f32-as-f64 values. json.Marshal([]float32) otherwise
			// uses the shortest f32 decimal, not the exact value used for acceptance.
			decoded := make([]float64, len(c.Truth))
			s := &stats[j]
			for slot, v := range r.Decrypted {
				decoded[slot] = float64(v)
				err := math.Abs(decoded[slot] - c.Truth[slot])
				if !finite(err) {
					t.Fatalf("%s trial %d slot %d: nonfinite decoded value or error", arms[j].name, i, slot)
				}
				s.LogicalSlotsChecked++
				s.Linf = math.Max(s.Linf, err)
				s.SumSquaredError += err * err
				if !finite(s.SumSquaredError) {
					t.Fatalf("%s trial %d slot %d: nonfinite squared error", arms[j].name, i, slot)
				}
				if err > tolerance {
					s.SlotsOverTolerance++
					if s.FirstFailure == nil {
						s.FirstFailure = &miss{i, slot, decoded[slot], c.Truth[slot], err}
					}
				}
			}
			check(t, writers[j].Encode(struct {
				observed
				Decoded []float64 `json:"decrypted"`
			}{r, decoded}))
		}
	}
	// Every finite miss is recorded before enforcing any numerical gate.
	for j, a := range arms {
		s := &stats[j]
		s.MSE = s.SumSquaredError / float64(s.LogicalSlotsChecked)
		s.PrecisionPassed = s.SlotsOverTolerance == 0
		s.Status = "PASS"
		if !s.PrecisionPassed {
			s.Status = "FAIL"
		}
		b, err := json.MarshalIndent(s, "", "  ")
		check(t, err)
		check(t, os.WriteFile(filepath.Join(dir, a.name, "accuracy.json"), b, 0644))
		msg := fmt.Sprintf("%s/%s NUMERICAL %s: %d/%d slots over %g; Linf %.10g MSE %.10g; hard_precision_gate=%t", work, a.name, s.Status, s.SlotsOverTolerance, s.LogicalSlotsChecked, tolerance, s.Linf, s.MSE, s.HardPrecisionGate)
		if f := s.FirstFailure; f != nil {
			msg += fmt.Sprintf("; first trial %d slot %d: decrypted %.15g, true %.15g, |error| %.10g", f.Trial, f.Slot, f.Decrypted, f.Truth, f.AbsError)
		}
		if !s.PrecisionPassed && s.HardPrecisionGate {
			t.Error(msg)
		} else {
			t.Log(msg)
		}
	}
}
