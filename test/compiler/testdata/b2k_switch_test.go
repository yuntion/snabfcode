// Independent exact integer check on the actual production Q/P and switch sites.
// No key material, noise observations or measurements enter a proof receipt.
package planned

import (
	"encoding/json"
	"math/big"
	"math/bits"
	"os"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/gadget"
)

func TestB2KSameKey(t *testing.T) {
	var graph struct {
		Width, Slots int
		Epochs       []struct {
			N, Scale int
			Q, P     []uint64
		}
		Instructions []struct {
			ID, Op             string
			Level, K, Rotation int
		}
	}
	buf, err := os.ReadFile(os.Getenv("B2K_PROGRAM"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(buf, &graph); err != nil || len(graph.Epochs) != 1 {
		t.Fatal("domain", err)
	}
	d := graph.Epochs[0]
	if len(d.P) != 1 {
		t.Fatal("not a singleton P")
	}
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: bits.TrailingZeros(uint(d.N)), Q: d.Q, P: d.P, LogDefaultScale: d.Scale})
	if err != nil {
		t.Fatal(err)
	}
	// Probe the actual conversion used by the diagnostic SDK at three adjacent
	// residues. The positive floor endpoint is not portable between conventions.
	tiny := p.RingQ().AtLevel(0)
	boundary := tiny.NewPoly()
	half := int64(d.Q[0] / 2)
	for j := range 3 {
		boundary.Coeffs[0][j] = uint64(half - 1 + int64(j))
	}
	converted := make([]*big.Int, d.N)
	for j := range converted {
		converted[j] = new(big.Int)
	}
	tiny.PolyToBigintCentered(boundary, 1, converted)
	for j, want := range []int64{half - 1, -half - 1, -half} {
		if converted[j].Cmp(big.NewInt(want)) != 0 {
			t.Fatalf("SDK CRT endpoint %d", j)
		}
	}
	kg := rlwe.NewKeyGenerator(p)
	sk, pk := kg.GenKeyPairNew()
	dec := rlwe.NewDecryptor(p, sk)
	seen := map[int]bool{}
	var gks []*rlwe.GaloisKey
	for _, i := range graph.Instructions {
		if (i.Op == "baby" || i.Op == "giant") && !seen[i.Rotation] {
			seen[i.Rotation] = true
			gks = append(gks, kg.GenGaloisKeyNew(p.GaloisElement(i.Rotation), sk))
		}
	}
	ev := ckks.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(kg.GenRelinearizationKeyNew(sk), gks...))
	pt := ckks.NewPlaintext(p, p.MaxLevel())
	pt.LogDimensions.Cols = bits.TrailingZeros(uint(graph.Slots))
	vals := make([]float64, graph.Slots)
	for j := range vals {
		vals[j] = float64((j%graph.Width)%7-3) / 8
	}
	if err = ckks.NewEncoder(p).Encode(vals, pt); err != nil {
		t.Fatal(err)
	}
	base, err := rlwe.NewEncryptor(p, pk).EncryptNew(pt)
	if err != nil {
		t.Fatal(err)
	}
	sites, coeffs, ctSquares := 0, 0, 0
	for _, i := range graph.Instructions {
		rel := i.Op == "relin"
		if !rel && i.Op != "baby" && i.Op != "giant" {
			continue
		}
		rq := p.RingQ().AtLevel(i.Level)
		in := base.CopyNew()
		in.Resize(1, i.Level)
		if rel {
			orig := dec.DecryptNew(in)
			in, err = ev.MulNew(in, in)
			if err != nil {
				t.Fatal(err)
			}
			want := rq.NewPoly()
			rq.MulCoeffsBarrett(orig.Value, orig.Value, want)
			if !rq.Equal(want, dec.DecryptNew(in).Value) {
				t.Fatal(i.ID, "degree-two same-source product")
			}
			ctSquares++
		}
		site := gadget.Site{ID: i.ID, Level: i.Level, K: i.K}
		var out *rlwe.Ciphertext
		// Giant K=0 mutates its argument. Preserve the genuine pre-call operand.
		operand := in.CopyNew()
		if rel {
			out, err = gadget.RelinearizeNew(ev, operand, site)
		} else if i.Op == "baby" {
			var group map[int]*rlwe.Ciphertext
			group, err = gadget.Babies(ev, operand, []int{i.Rotation}, map[int]gadget.Site{i.Rotation: site})
			out = group[i.Rotation]
		} else {
			out, err = gadget.Giant(ev, operand, i.Rotation, site)
		}
		if err != nil {
			t.Fatal(err)
		}
		// Rebuild the omitted term from the *original* c1/c2 prefix. Use the
		// same s or s^2 used for encryption; no access to gadget.project/digits.
		expected := dec.DecryptNew(in).Value
		if i.K == 1 {
			axis := in.Degree()
			tiny := rq.AtLevel(0)
			co := tiny.NewPoly()
			tiny.INTT(in.Value[axis], co)
			r := rq.NewPoly()
			q0 := d.Q[0]
			for j, v := range co.Coeffs[0] {
				v %= q0
				for l, q := range d.Q[:i.Level+1] {
					if v <= q0/2 {
						r.Coeffs[l][j] = v % q
					} else {
						r.Coeffs[l][j] = (q - (q0-v)%q) % q
					}
				}
			}
			rq.NTT(r, r)
			for j := 0; j < axis; j++ {
				rq.MulCoeffsMontgomery(r, sk.Value.Q, r)
			}
			rq.Sub(expected, r, expected)
		} else if i.K != 0 {
			t.Fatal("unsupported K")
		}
		if !rel {
			perm := rq.NewPoly()
			rq.AutomorphismNTT(expected, p.GaloisElement(i.Rotation), perm)
			expected = perm
		}
		// After stripping the signed cutoff, the remainder must fit the EVK-only
		// coefficient bound, not the much larger combined truncation envelope.
		diff := rq.NewPoly()
		rq.Sub(dec.DecryptNew(out).Value, expected, diff)
		rq.INTT(diff, diff)
		words := make([]*big.Int, d.N)
		for j := range words {
			words[j] = new(big.Int)
		}
		rq.PolyToBigintCentered(diff, 1, words)
		P := new(big.Int).SetUint64(d.P[0])
		ds := new(big.Int)
		for _, q := range d.Q[i.K : i.Level+1] {
			ds.Add(ds, new(big.Int).SetUint64(q))
		}
		bound := new(big.Int).Mul(ds, big.NewInt(int64(19*d.N)))
		hp := new(big.Int).Rsh(new(big.Int).Set(P), 1)
		bound.Add(bound, hp.Mul(hp, big.NewInt(int64(d.N+1))))
		bound.Quo(bound, P)
		for j, w := range words {
			if new(big.Int).Abs(w).Cmp(bound) > 0 {
				t.Fatalf("%s coefficient %d violates isolated EVK bound", i.ID, j)
			}
		}
		sites++
		coeffs += len(words)
	}
	if sites < 1 || ctSquares != 3 {
		t.Fatal("wrong graph switch inventory", sites, ctSquares)
	}
	t.Logf("sites=%d degree_two_squares=%d exact_centered_switch_coefficients=%d (one key, sampled corroboration only)", sites, ctSquares, coeffs)
}
