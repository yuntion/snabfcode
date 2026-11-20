package epochs

import (
	"math/big"
	"strings"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Metadata = RN128(2^0.5), effective = RN53(metadata) on embedDouble.
// Compare actual plaintext polynomials, not the display log or decoded samples.
func TestPhysicalEncodingReceipt(t *testing.T) {
	const metadata = "240615969168004511545033772477625056927/170141183460469231731687303715884105728"
	const asFloat = "6369051672525773/4503599627370496"
	for _, def := range []int{35, 60} {
		p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 10, LogQ: []int{48, 45}, LogP: []int{50}, LogDefaultScale: def})
		if err != nil {
			t.Fatal(err)
		}
		defaultScale := p.DefaultScale()
		base, _ := defaultScale.Value.Rat(nil)
		prec, eff := 53, asFloat
		if def > 53 {
			prec, eff = def, metadata
		}
		m, err := exactScale(metadata)
		if err != nil {
			t.Fatal(err)
		}
		out := p.DefaultScale().Mul(m)
		output, _ := out.Value.Rat(nil)
		values := []float64{1.25, -0.75}
		s := Program{Source: p.N(), Target: p.N(), Width: 2, Slots: 8, Fingerprint: strings.Repeat("ab", 32), Epochs: []Epoch{{N: p.N(), Scale: def, KeyID: "k", Q: p.Q(), P: p.P()}},
			Ports: []ExactPort{{ID: "in", N: p.N(), Level: 1, Output: base.RatString(), Call: "source", Metadata: base.RatString(), Effective: base.RatString(), Precision: prec, PayloadSHA256: payloadDigest(nil)},
				{ID: "pt", N: p.N(), Level: 1, Output: output.RatString(), Call: "coefficient", Metadata: metadata, Effective: eff, Precision: prec, Count: len(values), PayloadSHA256: payloadDigest(values)}},
			Plain: []PlainBinding{{Index: 1, N: p.N(), Level: 1, Values: values}}}
		eval := func(r *Runtime, in *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
			return r.Evaluator(p.N()).MulNew(in, r.Plain(1))
		}
		r, err := Bind(s, eval, RequireExactEncoding)
		if err != nil {
			t.Fatal(err)
		}
		pt := r.Plain(1)
		ref := ckks.NewPlaintext(p, 1)
		ref.LogDimensions = pt.LogDimensions
		ref.Scale = m
		_, _, enc := r.SourceEvaluation()
		tiled := make([]float64, 8)
		for i := range tiled {
			tiled[i] = values[i%2]
		}
		if err = enc.Encode(tiled, ref); err != nil || !ref.Value.Equal(&pt.Value) || !pt.Scale.Equal(m) {
			t.Fatalf("actual Encode mismatch: %v", err)
		}
		in, err := r.Encrypt([]float64{0.125, 0.5})
		if err != nil {
			t.Fatal(err)
		}
		ct, err := r.Eval(in)
		if err != nil || !ct.Scale.Equal(out) {
			t.Fatal("actual MulNew metadata mismatch", err)
		}
		// Change precisely the least significant metadata bit. LogScale cannot distinguish it.
		ulp := new(big.Rat).SetFrac(big.NewInt(1), new(big.Int).Lsh(big.NewInt(1), uint(127-def)))
		perturbed, err := exactScale(new(big.Rat).Add(output, ulp).RatString())
		if err != nil {
			t.Fatal(err)
		}
		ct.Scale = perturbed
		if r.Check("pt", ct, p.N(), 1, float64(def)+0.5) == nil {
			t.Fatal("accepted terminal 1-ulp drift")
		}
		wrong := s
		wrong.Ports = append([]ExactPort(nil), s.Ports...)
		wrong.Ports[1].Effective = "1"
		if _, err = Bind(wrong, eval, RequireExactEncoding); err == nil {
			t.Fatal("accepted wrong encoder branch")
		}
		wrong = s
		wrong.Plain = nil
		if _, err = Bind(wrong, eval, RequireExactEncoding); err == nil {
			t.Fatal("accepted missing plaintext")
		}
		wrong = s
		wrong.Ports = nil
		if _, err = Bind(wrong, eval, RequireExactEncoding); err == nil {
			t.Fatal("accepted absent receipt set")
		}
	}
}
