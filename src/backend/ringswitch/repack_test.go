package ringswitch

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend"
)

func splitParams(t *testing.T, logN int) (ckks.Parameters, ckks.Parameters) {
	t.Helper()
	src, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: logN, LogQ: []int{50, 40, 40}, LogP: []int{50}, LogDefaultScale: 40})
	if e != nil {
		t.Fatal(e)
	}
	dst, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: logN - 1, Q: src.Q()[:2], P: src.P(), LogDefaultScale: 40})
	if e != nil {
		t.Fatal(e)
	}
	return src, dst
}
func splitDecode(t *testing.T, p ckks.Parameters, key *rlwe.SecretKey, ct *rlwe.Ciphertext) []complex128 {
	t.Helper()
	x := make([]complex128, p.MaxSlots())
	if e := ckks.NewEncoder(p).Decode(rlwe.NewDecryptor(p, key).DecryptNew(ct), x); e != nil {
		t.Fatal(e)
	}
	return x
}
func splitClose(t *testing.T, a, b complex128) float64 {
	t.Helper()
	err := cmplx.Abs(a - b)
	if !(err < math.Exp2(-12)) {
		t.Fatalf("got %v want %v |difference|=%g", a, b, err)
	}
	return err
}

func TestFullSlotSplitI19(t *testing.T) {
	src, dst := splitParams(t, 14)
	S := dst.MaxSlots()
	level := src.MaxLevel()
	gen := rlwe.NewKeyGenerator(src)
	sk := gen.GenSecretKeyNew()
	smallSK := rlwe.NewKeyGenerator(dst).GenSecretKeyNew()
	keyLevel := level - 1
	down := gen.GenEvaluationKeyNew(sk, smallSK, rlwe.EvaluationKeyParameters{LevelQ: &keyLevel})
	fold := gen.GenGaloisKeyNew(src.GaloisElement(S), sk, rlwe.EvaluationKeyParameters{LevelQ: &keyLevel})
	raw, e := down.MarshalBinary()
	if e != nil {
		t.Fatal(e)
	}
	keyID := fmt.Sprintf("%x", sha256.Sum256(raw))
	sparse, e := backend.NewSparseDownConverter(src, dst, down)
	if e != nil {
		t.Fatal(e)
	}
	var mapID string
	for _, tc := range []struct {
		name string
		used int
	}{{"I19-and-boundary-impulses", 2 * S}, {"full-complex", 2 * S}, {"one-extra-slot", S + 1}, {"padded-tail", 2*S - 17}} {
		t.Run(tc.name, func(t *testing.T) {
			f, e := NewFullSlotSplitter(src, dst, level, tc.used, fold, down)
			if e != nil {
				t.Fatal(e)
			}
			x := make([]complex128, 2*S)
			if tc.name == "I19-and-boundary-impulses" {
				x[0], x[S], x[S-1], x[2*S-1] = .75, -.25, .125+.375i, -.25+.125i
			} else {
				rng := rand.New(rand.NewSource(20261120))
				for i := range tc.used {
					x[i] = complex(rng.Float64()-.5, rng.Float64()-.5)
				}
				x[0], x[S] = .75, -.25
			}
			pt := ckks.NewPlaintext(src, level)
			if e = ckks.NewEncoder(src).Encode(x, pt); e != nil {
				t.Fatal(e)
			}
			in, e := rlwe.NewEncryptor(src, sk).EncryptNew(pt)
			if e != nil {
				t.Fatal(e)
			}
			before := in.CopyNew()
			if _, e = sparse.Convert(in); e == nil {
				t.Fatal("old sparse DownConv accepted full independent slots")
			}
			if tc.name == "I19-and-boundary-impulses" {
				// Oracle only: directly trace the *unfolded* full input. Decode
				// metadata is assigned on the actual target output, not the input.
				proj := ckks.NewCiphertext(dst, 1, keyLevel)
				aligned := f.eval.DropLevelNew(in, 1)
				if e = rlwe.NewEvaluator(src, nil).ApplyEvaluationKey(aligned, down, proj); e != nil {
					t.Fatal(e)
				}
				proj.LogDimensions = dst.LogMaxDimensions()
				v := splitDecode(t, dst, smallSK, proj)
				for j, y := range v {
					splitClose(t, y, (x[j]+x[j+S])/2)
				}
				t.Logf("unfolded normalized trace of (.75,-.25): %.8f", real(v[0]))
				// Each folded producer is full-capacity in the source ring, with
				// two approximately equal lifts; metadata alone is not a proof.
				for half := range 2 {
					var r SplitObservation
					p, e := f.fold(in, half, &r)
					if e != nil {
						t.Fatal(e)
					}
					if _, e = sparse.Convert(p.ct); e == nil {
						t.Fatal("fold laundered metadata")
					}
					v := splitDecode(t, src, sk, p.ct)
					for j := range S {
						splitClose(t, v[j], x[half*S+j])
						splitClose(t, v[j+S], x[half*S+j])
					}
				}
			}
			out, r, e := f.Split(in)
			if e != nil {
				t.Fatal(e)
			}
			if !in.Equal(before) {
				t.Fatal("source was mutated")
			}
			if out[0] == out[1] || r.MaskMulCount != 2 || r.RescaleCount != 2 || r.FoldRotCount != 2 || r.DownConvCount != 2 || r.SwitchKeyBytes != len(raw) || r.FoldGaloisElement != src.GaloisElement(S) || len(r.SlotMapID) != 64 {
				t.Fatalf("incomplete physical observation: %+v", r)
			}
			if mapID != "" && mapID != r.SlotMapID {
				t.Fatal("map changed with data/padding")
			}
			mapID = r.SlotMapID
			var peak float64
			for h, ct := range out {
				z := r.Outputs[h]
				if z.DownKeySHA256 != keyID || z.N != dst.N() || z.Level != level-1 || math.Abs(z.LogScale-in.LogScale()) > 1e-8 || z.LogicalStart != h*S || z.UsedSlots != min(S, tc.used-h*S) || ct.LogDimensions != dst.LogMaxDimensions() {
					t.Fatalf("wrong target state: %+v", z)
				}
				for j, y := range splitDecode(t, dst, smallSK, ct) {
					peak = max(peak, splitClose(t, y, x[h*S+j]))
				}
			}
			t.Logf("N=%d->%d used=%d L=%d->%d all %d slots L_inf=%.3g down_key_bytes=%d", src.N(), dst.N(), tc.used, level, level-1, 2*S, peak, len(raw))
		})
	}
}

func TestFullSlotSplitRejectsUnboundShapes(t *testing.T) {
	src, dst := splitParams(t, 9)
	level := src.MaxLevel()
	S := dst.MaxSlots()
	gen := rlwe.NewKeyGenerator(src)
	sk := gen.GenSecretKeyNew()
	small := rlwe.NewKeyGenerator(dst).GenSecretKeyNew()
	good := gen.GenGaloisKeyNew(src.GaloisElement(S), sk)
	down := gen.GenEvaluationKeyNew(sk, small)
	bad := func(d ckks.Parameters, l, used int, gk *rlwe.GaloisKey, evk *rlwe.EvaluationKey) {
		t.Helper()
		if _, e := NewFullSlotSplitter(src, d, l, used, gk, evk); e == nil {
			t.Fatal("unsupported shape/key accepted")
		}
	}
	bad(dst, 0, 2*S, good, down)
	bad(dst, level, S, good, down)
	bad(dst, level, 2*S+1, good, down)
	bad(src, level, 2*S, good, down)
	bad(dst, level, 2*S, nil, down)
	bad(dst, level, 2*S, good, nil)
	bad(dst, level, 2*S, gen.GenGaloisKeyNew(src.GaloisElement(1), sk), down)
	tooSmall, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: src.LogN() - 2, Q: dst.Q(), P: dst.P(), LogDefaultScale: 40})
	if e != nil {
		t.Fatal(e)
	}
	bad(tooSmall, level, 2*S, good, down)
	z := 0
	bad(dst, level, 2*S, good, gen.GenEvaluationKeyNew(sk, small, rlwe.EvaluationKeyParameters{LevelQ: &z}))
	wrongP, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: dst.LogN(), Q: dst.Q(), LogP: []int{48}, LogDefaultScale: 40})
	if e != nil {
		t.Fatal(e)
	}
	bad(wrongP, level, 2*S, good, down)
	f, e := NewFullSlotSplitter(src, dst, level, 2*S, good, down)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = f.projectFolded(foldedHalf{}); e == nil {
		t.Fatal("missing fold producer accepted")
	}
	ct := ckks.NewCiphertext(src, 1, level)
	ct.LogDimensions = dst.LogMaxDimensions()
	r, obs, e := f.Split(ct)
	if e == nil || r[0] != nil || r[1] != nil || obs.DownConvCount != 0 {
		t.Fatal("forged smaller metadata accepted")
	}
	ct.LogDimensions = src.LogMaxDimensions()
	ct.IsBitReversed = true
	if _, _, e = f.Split(ct); e == nil {
		t.Fatal("unverified slot order accepted")
	}
	ct.IsBitReversed = false
	ct.Scale = rlwe.NewScale(0)
	if _, _, e = f.Split(ct); e == nil {
		t.Fatal("zero CKKS scale accepted")
	}
}
