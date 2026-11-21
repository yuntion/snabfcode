package main

import (
	"encoding/json"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math"
	"math/big"
	"os"
)

func must(err error) {
	if err != nil {
		panic(err)
	}
}
func rat(s rlwe.Scale) string { x, _ := s.Value.Rat(nil); return x.RatString() }
func first(p ckks.Parameters, l int, ntt bool, z ring.Poly) int64 {
	r := p.RingQ().AtLevel(l)
	if ntt {
		t := r.NewPoly()
		r.INTT(z, t)
		z = t
	}
	q := int64(r.SubRings[0].Modulus)
	v := int64(z.Coeffs[0][0] % uint64(q))
	if v > q/2 {
		v -= q
	}
	return v
}
func tile(n int, x float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = x
	}
	return v
}
func main() {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 4, LogQ: []int{57, 50}, LogP: []int{52}, LogDefaultScale: 40})
	must(err)
	ev := ckks.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(nil))
	out := map[string]any{}
	coeff := map[string]int64{}
	out["coeff0"] = coeff
	meta := rlwe.NewScale(uint64(1)<<53 | 1)
	out["meta"] = rat(meta)
	out["scale_prec"] = meta.Value.Prec()
	out["float64"] = new(big.Rat).SetFloat64(meta.Float64()).RatString()
	encode := func(k string, prec uint, batched bool, s rlwe.Scale, v any) {
		pt := ckks.NewPlaintext(p, p.MaxLevel())
		pt.IsBatched = batched
		pt.Scale = s
		must(ckks.NewEncoder(p, prec).Encode(v, pt))
		coeff[k] = first(p, pt.Level(), pt.IsNTT, pt.Value)
	}
	encode("batch_53_f64", 53, true, meta, tile(p.MaxSlots(), 1))
	encode("batch_60_f64", 60, true, meta, tile(p.MaxSlots(), 1))
	encode("coeff_60_f64", 60, false, meta, []float64{1})
	encode("coeff_53_big128", 53, false, meta, []*big.Float{new(big.Float).SetPrec(128).SetInt64(1)})
	encode("coeff_53_big53", 53, false, meta, []*big.Float{big.NewFloat(1)})
	encode("coeff_half", 53, false, rlwe.NewScale(1), []float64{0.5})
	encode("coeff_neg_half", 53, false, rlwe.NewScale(1), []float64{-0.5})
	encode("coeff_below_half", 53, false, rlwe.NewScale(1), []float64{math.Nextafter(0.5, 0)})
	encode("coeff_below_half_big", 53, false, rlwe.NewScale(1), []*big.Float{new(big.Float).SetPrec(128).SetFloat64(math.Nextafter(0.5, 0))})
	// Zero-noise evaluator carriers: actual encoded c0 and zero c1, not an encryption trial.
	carrier := func(s uint64, x float64) *rlwe.Ciphertext {
		pt := ckks.NewPlaintext(p, p.MaxLevel())
		pt.Scale = rlwe.NewScale(s)
		must(ckks.NewEncoder(p, 53).Encode(tile(p.MaxSlots(), x), pt))
		ct := ckks.NewCiphertext(p, 1, pt.Level())
		*ct.MetaData = *pt.MetaData
		ct.Value[0].Copy(pt.Value)
		return ct
	}
	join := func(a, b uint64, rev, sub bool) map[string]any {
		l, r := carrier(a, 1), carrier(b, 0)
		if rev {
			l, r = r, l
		}
		var o *rlwe.Ciphertext
		var e error
		if sub {
			o, e = ev.SubNew(l, r)
		} else {
			o, e = ev.AddNew(l, r)
		}
		must(e)
		return map[string]any{"out_metadata": rat(o.Scale), "coeff0": first(p, o.Level(), o.IsNTT, o.Value[0])}
	}
	a := uint64(1) << 40
	out["near_add"] = join(a, a+1, false, false)
	out["near_sub_reverse"] = join(a, a+1, true, true)
	out["four_seven"] = join(4, 7, false, false)
	out["four_nine"] = join(4, 9, false, false)
	ratio := rlwe.NewScale(7).Div(rlwe.NewScale(4))
	m, _ := ratio.Value.Int(nil)
	out["alignment_trunc"] = m.String()
	out["scale_bigint"] = ratio.BigInt().String()
	h, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 4, Q: []uint64{97, 353, 449}, P: []uint64{193}, LogDefaultScale: 80})
	must(err)
	he := ckks.NewEvaluator(h, rlwe.NewMemEvaluationKeySet(nil))
	ct := ckks.NewCiphertext(h, 1, 2)
	ct.Scale = rlwe.NewScale(65536)
	rs := ckks.NewCiphertext(h, 1, 2)
	must(he.Rescale(ct, rs))
	drop := he.DropLevelNew(ct, 2)
	out["rescale"] = map[string]any{"q": h.Q(), "before": rat(ct.Scale), "after": rat(rs.Scale), "after_level": rs.Level(), "drop_scale": rat(drop.Scale), "drop_level": drop.Level(), "one_div": rat(ct.Scale.Div(rlwe.NewScale(449 * 353)))}
	s := rlwe.NewScale(new(big.Float).SetPrec(128).SetRat(big.NewRat(1, 3)))
	out["one_third_as_scale"] = rat(s)
	req := big.NewRat(97, 3)
	ptS := rlwe.NewScale(new(big.Float).SetPrec(128).SetRat(req))
	out["fold_requested"] = req.RatString()
	out["fold_actual_meta"] = rat(ptS)
	out["fold_f64_eff"] = new(big.Rat).SetFloat64(ptS.Float64()).RatString()
	j := json.NewEncoder(os.Stdout)
	j.SetIndent("", "  ")
	must(j.Encode(out))
}
