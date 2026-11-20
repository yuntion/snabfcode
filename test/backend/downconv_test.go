package backend_test

import (
	"math/cmplx"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend"
)

func parameters(t *testing.T) (ckks.Parameters, ckks.Parameters) {
	t.Helper()
	large, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: 14, LogQ: []int{50, 40, 40}, LogP: []int{50}, LogDefaultScale: 40,
	})
	if err != nil {
		t.Fatal(err)
	}
	small, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: 13, Q: large.Q()[:2], P: large.P(), LogDefaultScale: 40,
	})
	if err != nil {
		t.Fatal(err)
	}
	return large, small
}

func encrypt(t *testing.T, p ckks.Parameters, key *rlwe.SecretKey, level, logSlots int, x []complex128) *rlwe.Ciphertext {
	t.Helper()
	pt := ckks.NewPlaintext(p, level)
	pt.LogDimensions.Cols = logSlots
	if err := ckks.NewEncoder(p).Encode(x, pt); err != nil {
		t.Fatal(err)
	}
	ct, err := rlwe.NewEncryptor(p, key).EncryptNew(pt)
	if err != nil {
		t.Fatal(err)
	}
	return ct
}

func decode(t *testing.T, p ckks.Parameters, key *rlwe.SecretKey, ct *rlwe.Ciphertext, n int) []complex128 {
	t.Helper()
	x := make([]complex128, n)
	if err := ckks.NewEncoder(p).Decode(rlwe.NewDecryptor(p, key).DecryptNew(ct), x); err != nil {
		t.Fatal(err)
	}
	return x
}

func TestSparseDownConv(t *testing.T) {
	large, small := parameters(t)
	g := rlwe.NewKeyGenerator(large)
	srcKey := g.GenSecretKeyNew()
	dstKey := rlwe.NewKeyGenerator(small).GenSecretKeyNew()
	level := small.MaxLevel()
	key := g.GenEvaluationKeyNew(srcKey, dstKey, rlwe.EvaluationKeyParameters{LevelQ: &level})
	conv, err := backend.NewSparseDownConverter(large, small, key)
	if err != nil {
		t.Fatal(err)
	}
	x := []complex128{0.75, -0.33, 0.21 + 0.19i, -0.11i, 0.5, -0.9, 0.98, 0.071}
	source := encrypt(t, large, srcKey, level, 3, x)
	out, err := conv.Convert(source)
	if err != nil {
		t.Fatal(err)
	}
	if out.Value[0].N() != small.N() || out.Level() != source.Level() ||
		out.Scale.Cmp(source.Scale) != 0 || out.LogDimensions != source.LogDimensions {
		t.Fatal("DownConv did not preserve its level, scale, and layout contract")
	}
	y := decode(t, small, dstKey, out, len(x))
	before := decode(t, large, srcKey, source, len(x))
	var maxMapError, maxInputError float64
	for i := range x {
		maxInputError = max(maxInputError, cmplx.Abs(y[i]-x[i]))
		maxMapError = max(maxMapError, cmplx.Abs(y[i]-before[i]))
	}
	if !(maxInputError <= 1e-6 && maxMapError <= 1e-6) {
		t.Fatalf("DownConv errors: original=%g, map=%g", maxInputError, maxMapError)
	}
	t.Logf("DownConv N=%d->%d, slots=%d, level=%d, log2(scale)=%.1f, map L_inf=%g, key bytes=%d",
		large.N(), small.N(), len(x), out.Level(), out.LogScale(), maxMapError, key.BinarySize())

	// Real full packing with unequal values in each trace orbit: the planner
	// cannot reinterpret this input as having only N'/2 logical slots.
	full := make([]complex128, large.MaxSlots())
	for i := range small.MaxSlots() {
		full[i], full[i+small.MaxSlots()] = 0.75, -0.25
	}
	fullCt := encrypt(t, large, srcKey, level, large.LogMaxSlots(), full)
	if _, err := conv.Convert(fullCt); err == nil {
		t.Fatal("independent slots were silently merged")
	}
	// Check the semantics of the native projection itself. These metadata
	// edits apply only in the oracle, to decode the projected slot count.
	projected := ckks.NewCiphertext(small, 1, level)
	if err := rlwe.NewEvaluator(large, nil).ApplyEvaluationKey(fullCt, key, projected); err != nil {
		t.Fatal(err)
	}
	projected.LogDimensions.Cols = small.LogMaxSlots()
	for i, v := range decode(t, small, dstKey, projected, small.MaxSlots()) {
		if !(cmplx.Abs(v-0.25) <= 1e-6) {
			t.Fatalf("trace orbit %d did not average: got %v, want 0.25", i, v)
		}
	}
	tooHigh := encrypt(t, large, srcKey, large.MaxLevel(), 3, x)
	if _, err := conv.Convert(tooHigh); err == nil {
		t.Fatal("conversion silently dropped a Q limb")
	}
}
