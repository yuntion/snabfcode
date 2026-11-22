// Compiled beside the freshly emitted model.go, never a replacement runtime.
package planned

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
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
	_, p, _ := r.SourceEvaluation()
	count := 0
	for _, tr := range audit.B2I.Transfers {
		if tr.Formula != "b2i/constant-real-tile-v6.2.0" {
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
		found := false
		for i, port := range r.Program.Ports {
			if port.ID != id {
				continue
			}
			pt := r.Plain(i)
			if found || pt == nil || !pt.IsNTT || !pt.IsBatched || pt.Slots() != r.Program.Slots {
				t.Fatal("not the planned batched plaintext", id)
			}
			found = true
			ringQ := p.RingQ().AtLevel(pt.Level())
			out := ringQ.NewPoly()
			ringQ.INTT(pt.Value, out)
			if pt.IsMontgomery {
				ringQ.IMForm(out, out)
			}
			for l, limb := range out.Coeffs {
				q := int64(p.Q()[l])
				for j, got := range limb {
					var want uint64
					if j == 0 {
						want = uint64((c%q + q) % q)
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
		count++
	}
	if count != 1 {
		t.Fatalf("expected one proved scalar on this graph, got %d", count)
	}
}
