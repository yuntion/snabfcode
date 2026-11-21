package planned

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math"
	"testing"
)

func TestSharedProduct(t *testing.T) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 10, LogQ: []int{46, 40, 40}, LogP: []int{45}, LogDefaultScale: 35})
	if err != nil {
		t.Fatal(err)
	}
	kg := rlwe.NewKeyGenerator(p)
	sk, pk := kg.GenKeyPairNew()
	enc := ckks.NewEncoder(p)
	pt := ckks.NewPlaintext(p, p.MaxLevel())
	values := make([]float64, p.MaxSlots())
	for i := range values {
		values[i] = float64(i%13-6) / 16
	}
	if err = enc.Encode(values, pt); err != nil {
		t.Fatal(err)
	}
	x, err := rlwe.NewEncryptor(p, pk).EncryptNew(pt)
	if err != nil {
		t.Fatal(err)
	}
	before := x.CopyNew()
	e := ckks.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(kg.GenRelinearizationKeyNew(sk)))
	out, err := evaluate(e, x)
	if err != nil {
		t.Fatal(err)
	}
	if !before.Equal(x) || out.Degree() != 1 {
		t.Fatal("SSA input mutated or unexpected output degree")
	}
	decoded := make([]float64, p.MaxSlots())
	if err = enc.Decode(rlwe.NewDecryptor(p, sk).DecryptNew(out), decoded); err != nil {
		t.Fatal(err)
	}
	for i, v := range decoded {
		if !(math.Abs(v-2*values[i]*values[i]) < 1.0/4096) {
			t.Fatalf("slot %d: %g", i, v)
		}
	}
}
