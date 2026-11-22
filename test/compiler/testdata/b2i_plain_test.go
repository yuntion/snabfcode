// Compiled beside the freshly emitted model.go, never a replacement runtime.
package planned

import (
	"encoding/json"
	"math/bits"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestB2IActualBoundPlaintext(t *testing.T) {
	var audit struct {
		Seal string `json:"emit_sha256"`
		B2I  struct {
			Transfers []struct {
				Site    string  `json:"site_id"`
				Formula string  `json:"formula_ref"`
				C       *string `json:"encoded_constant"`
			} `json:"transfers"`
		} `json:"b2i_local"`
	}
	b, err := os.ReadFile(os.Getenv("B2I_AUDIT"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &audit); err != nil || audit.Seal != PlanSHA256 {
		t.Fatal("audit not bound to emitted runtime", err)
	}
	r, err := New()
	if err != nil {
		t.Fatal(err)
	}
	eval, p, enc := r.SourceEvaluation()
	wantIDs := map[string]bool{"poly0.n2.pt0": true, "poly0.n2.constant": true, "poly0.n4.pt0": true}
	for _, tr := range audit.B2I.Transfers {
		if tr.Formula != "b2i/constant-real-tile-v6.2.0" && tr.Formula != "b2i/dc-word-bounded-ntt-v6.2.0" {
			continue
		}
		if tr.C == nil {
			t.Fatal("absent centered coefficient", tr.Site)
		}
		c, err := strconv.ParseInt(*tr.C, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		id := strings.TrimSuffix(strings.TrimPrefix(tr.Site, "node/"), "/plain_encoding_at_output")
		if !wantIDs[id] {
			t.Fatal("unexpected or duplicate DC proof", id)
		}
		delete(wantIDs, id)
		found := false
		for i, port := range r.Program.Ports {
			if port.ID != id {
				continue
			}
			pt := r.Plain(i)
			if found || pt == nil || !pt.IsNTT || pt.IsMontgomery || !pt.IsBatched || pt.Slots() != r.Program.Slots {
				t.Fatal("not the planned batched plaintext", id)
			}
			found = true
			ringQ := p.RingQ().AtLevel(pt.Level())
			out := ringQ.NewPoly()
			ringQ.INTT(pt.Value, out)
			for l, limb := range out.Coeffs {
				q := int64(p.Q()[l])
				dc := uint64((c%q + q) % q)
				for j, got := range limb {
					if pt.Value.Coeffs[l][j] != dc {
						t.Fatalf("%s noncanonical or wrong NTT word at %d:%d", id, l, j)
					}
					var want uint64
					if j == 0 {
						want = dc
					}
					if got != want {
						t.Fatalf("%s limb %d coefficient %d: %d != %d", id, l, j, got, want)
					}
				}
			}
			t.Logf("%s C=%d; %d limbs x %d coefficients", id, c, pt.Level()+1, p.N())
		}
		if !found {
			t.Fatal("missing runtime encoding call", id)
		}
	}
	if len(wantIDs) != 0 {
		t.Fatal("missing DC proofs", wantIDs)
	}
	// Isolate the ring identity in the actual evaluator. This constructed
	// (K,0) ciphertext is not a sample from an encryption distribution.
	q := int64(p.Q()[0])
	for _, slots := range []int{r.Program.Slots, p.MaxSlots()} {
		for _, w := range []float64{5, -5, -0.25} {
			pt := ckks.NewPlaintext(p, 0)
			pt.Scale = rlwe.NewScale(1)
			pt.LogDimensions.Cols = bits.TrailingZeros(uint(slots))
			v := make([]float64, slots)
			for j := range v {
				v[j] = w
			}
			if err := enc.Encode(v, pt); err != nil {
				t.Fatal(err)
			}
			in := ckks.NewCiphertext(p, 1, 0)
			*in.MetaData = *pt.MetaData
			for j := range in.Value[0].Coeffs[0] {
				in.Value[0].Coeffs[0][j] = uint64(q / 2)
			}
			out, err := eval.MulNew(in, pt)
			if err != nil {
				t.Fatal(err)
			}
			// For these w only, int64(w) equals the rounded multiplier.
			before := (q / 2) * int64(w)
			center := (before%q + q) % q
			if 2*center > q {
				center -= q
			}
			if w != -0.25 && (center-before == 0 || (center-before)%q != 0) {
				t.Fatal("expected a real centering correction")
			}
			ringQ := p.RingQ().AtLevel(0)
			for k, poly := range out.Value {
				coeff := ringQ.NewPoly()
				ringQ.INTT(poly, coeff)
				for j, word := range coeff.Coeffs[0] {
					var want uint64
					if j == 0 && k == 0 {
						want = uint64((center + q) % q)
					}
					if word != want {
						t.Fatalf("slots=%d w=%g component=%d coefficient=%d: %d != %d", slots, w, k, j, word, want)
					}
				}
			}
		}
	}
}
