package backend_test

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math"
	"snabf.local/compiler/src/backend/gadget"
	"testing"
)

func TestCenteredGadget(t *testing.T) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 10, LogQ: []int{21, 32, 32, 32}, LogP: []int{48}, LogDefaultScale: 32})
	if err != nil {
		t.Fatal(err)
	}
	gen := rlwe.NewKeyGenerator(p)
	sk := gen.GenSecretKeyNew()
	e := ckks.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(gen.GenRelinearizationKeyNew(sk), gen.GenGaloisKeyNew(p.GaloisElement(1), sk)))
	enc := ckks.NewEncoder(p)
	pt := ckks.NewPlaintext(p, 3)
	if err = enc.Encode([]float64{.25, -.5}, pt); err != nil {
		t.Fatal(err)
	}
	ct, err := rlwe.NewEncryptor(p, sk).EncryptNew(pt)
	if err != nil {
		t.Fatal(err)
	}
	mul, err := e.MulNew(ct, ct)
	if err != nil {
		t.Fatal(err)
	}
	natural, err := e.RelinearizeNew(mul)
	if err != nil {
		t.Fatal(err)
	}
	events := []gadget.Event{}
	gadget.Observe = func(ev gadget.Event) { events = append(events, ev) }
	defer func() { gadget.Observe = nil }()
	exact, err := gadget.RelinearizeNew(e, mul, gadget.Site{Level: 3})
	if err != nil || !exact.Equal(natural) || len(events) != 0 {
		t.Fatal("native k=0 identity", err)
	}
	approx, err := gadget.RelinearizeNew(e, mul, gadget.Site{Level: 3, K: 1})
	if err != nil {
		t.Fatal(err)
	}
	decode := func(c *rlwe.Ciphertext) []float64 {
		v := make([]float64, p.MaxSlots())
		if err := enc.Decode(rlwe.NewDecryptor(p, sk).DecryptNew(c), v); err != nil {
			t.Fatal(err)
		}
		return v
	}
	a, b := decode(approx), decode(exact)
	var peak float64
	for j := range a {
		peak = math.Max(peak, math.Abs(a[j]-b[j]))
	}
	if len(events) != 1 || events[0].RowsSkipped != 1 || events[0].RowsExecuted != 3 || events[0].ResidualMax > float64(p.Q()[0]/2) || peak == 0 || peak > 1e-5 {
		t.Fatalf("residual %.4g, events %+v", peak, events)
	}
	rot, err := gadget.Giant(e, natural, 1, gadget.Site{Level: 3, K: 1})
	if err != nil {
		t.Fatal(err)
	}
	native, err := e.RotateNew(natural, 1)
	if err != nil {
		t.Fatal(err)
	}
	a, b = decode(rot), decode(native)
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-5 {
			t.Fatal("rotation error")
		}
	}
	if _, err = gadget.RelinearizeNew(e, mul, gadget.Site{Level: 2, K: 1}); err == nil {
		t.Fatal("stale site accepted")
	}
	t.Logf("paired relin slot Linf=%g, event=%+v", peak, events[0])
}
