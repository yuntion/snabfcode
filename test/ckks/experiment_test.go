package ckks_test

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"testing"
	"time"

	"snabf.local/compiler/build/baseline"
)

func matvec(w [4][4]float64, x [4]float64) (y [4]float64) {
	for i := range w {
		for j := range x {
			y[i] += w[i][j] * x[j]
		}
	}
	return
}

// Independent float64 reference for test/fixtures/linear_silu_linear.mlir.
func reference(x [4]float64) [4]float64 {
	h := matvec([4][4]float64{
		{0.50, -0.25, 0.125, 0.0},
		{0.0, 0.375, 0.125, 0.25},
		{-0.25, 0.0, 0.50, 0.125},
		{0.125, -0.125, 0.25, 0.25},
	}, x)
	for i, v := range h {
		h[i] = v / (1 + math.Exp(-v))
	}
	return matvec([4][4]float64{
		{0.25, 0.50, 0.0, -0.125},
		{0.375, -0.125, 0.25, 0.0},
		{0.0, 0.25, -0.25, 0.50},
		{-0.125, 0.0, 0.50, 0.25},
	}, h)
}

func TestLinearSiLULinear(t *testing.T) {
	const tolerance = 1.0 / (1 << 12)
	const cornerCount, randomCount = 16, 48
	const trials = cornerCount + randomCount
	eval, params, encoder, enc, dec := baseline.Linear_silu_linear__configure()
	data, err := os.ReadFile("../fixtures/reference_profile.json")
	if err != nil {
		t.Fatal(err)
	}
	var profile struct {
		LogN, LogDefaultScale int
		Q, P                  []uint64
	}
	if err = json.Unmarshal(data, &profile); err != nil {
		t.Fatal(err)
	}
	if params.LogN() != profile.LogN || params.LogDefaultScale() != profile.LogDefaultScale ||
		!slices.Equal(params.Q(), profile.Q) || !slices.Equal(params.P(), profile.P) {
		t.Fatal("generated CKKS runtime diverged from the frozen reference profile")
	}
	linearTransforms, plaintexts := baseline.Linear_silu_linear__preprocessing(params, encoder)
	if len(linearTransforms) != 2 {
		t.Fatalf("expected two encrypted linear transforms, got %d", len(linearTransforms))
	}
	rng := rand.New(rand.NewPCG(314159, 271828))
	var maxError, squaredError, inputScale, outputScale float64
	var inputLevel, outputLevel int
	var trialsPassing, outsideTolerance int
	var evaluateTime time.Duration
	for trial := range trials {
		arg := make([]float32, 4)
		var x [4]float64
		for i := range arg {
			if trial < cornerCount {
				arg[i] = float32(2*((trial>>i)&1) - 1)
			} else {
				arg[i] = float32(2*rng.Float64() - 1)
			}
			x[i] = float64(arg[i])
		}
		plain := reference(x)
		ct := baseline.Linear_silu_linear__encrypt__arg0(eval, params, encoder, enc, arg)
		if len(ct) != 1 {
			t.Fatalf("input has %d ciphertexts; expected one", len(ct))
		}
		startLevel, startScale := ct[0].Level(), ct[0].LogScale()
		start := time.Now()
		out := baseline.Linear_silu_linear__preprocessed(eval, params, encoder, ct, linearTransforms, plaintexts)
		evaluateTime += time.Since(start)
		if len(out) != 1 {
			t.Fatalf("output has %d ciphertexts; expected one", len(out))
		}
		endLevel, endScale := out[0].Level(), out[0].LogScale()
		if trial == 0 {
			inputLevel, outputLevel, inputScale, outputScale = startLevel, endLevel, startScale, endScale
		} else if startLevel != inputLevel || endLevel != outputLevel ||
			startScale != inputScale || endScale != outputScale {
			t.Fatalf("runtime state changed at trial %d", trial)
		}
		got := baseline.Linear_silu_linear__decrypt__result0(eval, params, encoder, dec, out)
		if len(got) != len(plain) {
			t.Fatalf("decoded length %d, expected %d", len(got), len(plain))
		}
		pass := true
		for i, expected := range plain {
			absError := math.Abs(float64(got[i]) - expected)
			if math.IsNaN(absError) || math.IsInf(absError, 0) {
				t.Fatalf("nonfinite decoded output at trial %d slot %d", trial, i)
			}
			if absError > tolerance {
				pass = false
				outsideTolerance++
				if outsideTolerance <= 5 {
					t.Logf("trial %d slot %d: got %.9g, want %.9g (error %.3g)", trial, i, got[i], expected, absError)
				}
			}
			maxError = math.Max(maxError, absError)
			squaredError += absError * absError
		}
		if pass {
			trialsPassing++
		}
	}
	if outputLevel >= inputLevel {
		t.Fatalf("expected the circuit to consume levels, got %d -> %d", inputLevel, outputLevel)
	}
	stats, err := json.Marshal(map[string]any{
		"test": "heir_lattigo_reference", "trials": trials, "outputs_checked": trials * 4,
		"trials_passing": trialsPassing, "outputs_outside_tolerance": outsideTolerance,
		"linf": maxError, "mse": squaredError / (trials * 4), "tolerance": tolerance,
		"ring_n": params.N(), "q": params.Q(), "p": params.P(), "log_qp": params.LogQP(),
		"input_level": inputLevel, "output_level": outputLevel,
		"input_log2_scale": inputScale, "output_log2_scale": outputScale,
		"mean_evaluation_ms": float64(evaluateTime) / float64(time.Millisecond) / trials,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SNABF_METRICS %s", stats)
	if outsideTolerance > 0 {
		t.Errorf("%d values were outside tolerance in %d of %d trials", outsideTolerance, trials-trialsPassing, trials)
	}
}
