// Built beside an opt-in, freshly generated model.go on the integration host.
package planned

import (
	"encoding/json"
	"math"
	"math/big"
	"math/bits"
	"os"
	"strings"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestB2JFresh(t *testing.T) {
	var graph struct {
		Width, Slots int
		Seal         string `json:"emit_sha256"`
		Epochs       []struct {
			N, Scale int
			Q, P     []uint64
		}
		Instructions []struct {
			ID, Op string
			Level  int
			Values []float64
		}
		Ports struct {
			Receipts []struct {
				Index    int
				Metadata string `json:"metadata_scale"`
			}
		} `json:"encoding_receipts"`
	}
	b, err := os.ReadFile(os.Getenv("B2J_PROGRAM"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &graph); err != nil || PlanSHA256 != graph.Seal {
		t.Fatal("wrong generated plan", err)
	}
	m, err := New()
	if err != nil {
		t.Fatal(err)
	}
	// The old target has the very same seal but no sealed fresh entry.
	f, ok := any(m).(interface {
		EvalFresh([]float64) (*rlwe.Ciphertext, error)
	})
	if !ok {
		t.Fatal("model has no fresh entry")
	}
	if _, open := any(m).(interface {
		Eval(*rlwe.Ciphertext) (*rlwe.Ciphertext, error)
	}); open {
		t.Fatal("accepts external ciphertext")
	}
	x := make([]float64, graph.Width)
	for _, v := range []float64{math.Nextafter(1, 2), math.NaN(), math.Inf(-1)} {
		x[0] = v
		if _, e := f.EvalFresh(x); e == nil {
			t.Fatal("accepted out-of-domain input")
		}
	}
	if _, e := f.EvalFresh(x[:len(x)-1]); e == nil {
		t.Fatal("accepted wrong width")
	}
	for j := range x {
		x[j] = float64((5*j)%13-6) / 8
	}
	x[0] = 1
	for _, v := range []float64{1, -1} {
		x[0] = v
		ct, e := f.EvalFresh(x)
		if e != nil || ct.Degree() != 1 || ct.Level() != graph.Instructions[len(graph.Instructions)-1].Level {
			t.Fatal("fresh PK evaluation", e)
		}
	}
	decoder, ok := any(m).(interface {
		EvalFreshDecoded([]float64) ([]float64, error)
	})
	if !ok {
		t.Fatal("missing guarded output")
	}
	y, e := decoder.EvalFreshDecoded(x)
	if e != nil || len(y) != graph.Slots {
		t.Fatal("guarded decode", e)
	}
	for _, v := range y {
		if math.IsInf(v, 0) || math.IsNaN(v) {
			t.Fatal("nonfinite engineering output")
		}
	}
	xe, xs := rlwe.DefaultXe, rlwe.DefaultXs
	t.Cleanup(func() { rlwe.DefaultXe = xe; rlwe.DefaultXs = xs })
	rlwe.DefaultXe = ring.DiscreteGaussian{Sigma: 3.2, Bound: 20.1}
	if _, e := New(); e == nil || !strings.Contains(e.Error(), "sampler profile") {
		t.Fatal("wrong noise support", e)
	}
	rlwe.DefaultXe = xe
	rlwe.DefaultXs = ring.Ternary{H: 3}
	if _, e := New(); e == nil || !strings.Contains(e.Error(), "sampler profile") {
		t.Fatal("wrong secret support", e)
	}
	rlwe.DefaultXs = xs
	// Same concrete basis and actual evaluator/encoder. Synthetic degree-one
	// ciphertexts exercise universal claims at the boundary, not a noise law.
	d := graph.Epochs[0]
	p, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: bits.TrailingZeros(uint(d.N)), Q: d.Q, P: d.P, LogDefaultScale: d.Scale})
	if e != nil {
		t.Fatal(e)
	}
	eval, enc := ckks.NewEvaluator(p, nil), ckks.NewEncoder(p)
	one := big.NewInt(1)
	integer := func(q uint64) *big.Int { return new(big.Int).SetUint64(q) }
	mod := func(x *big.Int, q uint64) uint64 { return new(big.Int).Mod(x, integer(q)).Uint64() }
	constant := func(ct *rlwe.Ciphertext, j int, x *big.Int) {
		rq := p.RingQ().AtLevel(ct.Level())
		for l, v := range ct.Value[j].Coeffs {
			v[0] = mod(x, d.Q[l])
		}
		rq.NTT(ct.Value[j], ct.Value[j])
	}
	scales := map[int]*big.Float{}
	for _, v := range graph.Ports.Receipts {
		q, ok := new(big.Rat).SetString(v.Metadata)
		if !ok {
			t.Fatal("bad scale")
		}
		scales[v.Index] = new(big.Float).SetPrec(128).SetRat(q)
	}
	masks, limbs := 0, 0
	for i, ins := range graph.Instructions {
		if ins.Op != "mask" {
			continue
		}
		rq := p.RingQ().AtLevel(ins.Level)
		Q := rq.ModulusAtLevel[ins.Level]
		half := new(big.Int).Rsh(new(big.Int).Set(Q), 1)
		pt := ckks.NewPlaintext(p, ins.Level)
		pt.LogDimensions.Cols = bits.TrailingZeros(uint(graph.Slots))
		pt.Scale = rlwe.NewScale(scales[i])
		values := make([]float64, graph.Slots)
		for j := range values {
			values[j] = ins.Values[j%len(ins.Values)]
		}
		if e := enc.Encode(values, pt); e != nil {
			t.Fatal(e)
		}
		co := rq.NewPoly()
		rq.INTT(pt.Value, co)
		words := make([]*big.Int, d.N)
		for j := range words {
			words[j] = new(big.Int)
		}
		rq.PolyToBigintCentered(co, 1, words)
		j := 0
		for j < len(words) && new(big.Int).Abs(words[j]).Cmp(one) <= 0 {
			j++
		}
		if j == len(words) {
			t.Fatal("mask has no wrap witness", ins.ID)
		}
		ct := ckks.NewCiphertext(p, 1, ins.Level)
		ct.LogDimensions = pt.LogDimensions
		constant(ct, 0, half)
		got, e := eval.MulNew(ct, pt)
		if e != nil {
			t.Fatal(e)
		}
		rq.INTT(got.Value[0], co)
		expect := new(big.Int).Mul(half, words[j])
		if new(big.Int).Abs(expect).Cmp(half) <= 0 {
			t.Fatal("no actual product wrap")
		}
		for l, v := range co.Coeffs {
			if v[j] != mod(expect, d.Q[l]) {
				t.Fatal("wrong wrapped limb", ins.ID, l, j)
			}
			limbs++
		}
		masks++
	}
	// s(X)=1, valid ternary support. v=floor(Qin/2), but the two
	// centered residues sum to -(t+1)/2, so the output wraps by Qout.
	level := p.MaxLevel()
	rq := p.RingQ().AtLevel(level)
	q := d.Q[level]
	small := rq.AtLevel(level - 1)
	v := new(big.Int).Rsh(new(big.Int).Set(rq.ModulusAtLevel[level]), 1)
	c0 := new(big.Int).Neg(integer((q + 1) / 4))
	c1 := new(big.Int).Sub(v, c0)
	ct := ckks.NewCiphertext(p, 1, level)
	constant(ct, 0, c0)
	constant(ct, 1, c1)
	out := ckks.NewCiphertext(p, 1, level-1)
	if e := eval.Rescale(ct, out); e != nil {
		t.Fatal(e)
	}
	sum := small.NewPoly()
	small.Add(out.Value[0], out.Value[1], sum)
	small.INTT(sum, sum)
	centered := new(big.Int).Neg(new(big.Int).Rsh(new(big.Int).Set(small.ModulusAtLevel[level-1]), 1))
	for l, limb := range sum.Coeffs {
		if limb[0] != mod(centered, d.Q[l]) {
			t.Fatal("missed real rescale wrap", l)
		}
		for _, a := range limb[1:] {
			if a != 0 {
				t.Fatal("nonconstant wrap")
			}
		}
	}
	// All secret coefficients +1 and component residues +h give exactly
	// (N+1)h at coefficient N-1. Componentwise Rescale returns zero here.
	ct = ckks.NewCiphertext(p, 1, level)
	h := q / 2
	for l := 0; l <= level; l++ {
		ct.Value[0].Coeffs[l][d.N-1] = h % d.Q[l]
		for j := range ct.Value[1].Coeffs[l] {
			ct.Value[1].Coeffs[l][j] = h % d.Q[l]
		}
	}
	for _, poly := range ct.Value {
		rq.NTT(poly, poly)
	}
	if e := eval.Rescale(ct, out); e != nil {
		t.Fatal(e)
	}
	for _, poly := range out.Value {
		small.INTT(poly, sum)
		for _, limb := range sum.Coeffs {
			for _, a := range limb {
				if a != 0 {
					t.Fatal("wrong same-key rounding witness")
				}
			}
		}
	}
	t.Logf("fresh evaluations=3 (one decoded); actual wrapped diagonals=%d; checked active limbs=%d; two same-key rescale witnesses", masks, limbs)
}
