package gadget

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type fixture struct {
	p        ckks.Parameters
	e        *ckks.Evaluator
	dec      *rlwe.Decryptor
	encoder  *ckks.Encoder
	mul, rot *rlwe.Ciphertext
	truth    []float64
	calls    *transformCounts
}

func encryptedFixture(t *testing.T, alpha, level int) fixture {
	t.Helper()
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 8, LogQ: []int{21, 20, 58, 35, 39, 40, 32}, LogP: []int{59, 56, 54, 53}[:alpha], LogDefaultScale: 45})
	if err != nil {
		t.Fatal(err)
	}
	g := rlwe.NewKeyGenerator(p)
	sk := g.GenSecretKeyNew()
	keys := rlwe.NewMemEvaluationKeySet(g.GenRelinearizationKeyNew(sk), g.GenGaloisKeyNew(p.GaloisElement(1), sk), g.GenGaloisKeyNew(p.GaloisElement(2), sk), g.GenGaloisKeyNew(p.GaloisElement(3), sk))
	e := ckks.NewEvaluator(p, keys)
	enc := ckks.NewEncoder(p)
	pt := ckks.NewPlaintext(p, level)
	x := make([]float64, p.MaxSlots())
	for i := range x {
		x[i] = float64(i%59-29) / 64
	}
	if err = enc.Encode(x, pt); err != nil {
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
	rot, err := e.RelinearizeNew(mul)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x {
		x[i] *= x[i]
	}
	f := fixture{p, e, rlwe.NewDecryptor(p, sk), enc, mul, rot, x, new(transformCounts)}
	// Independent whole-operation counter, including native ModDown on BOTH
	// arms. This is not the per-event counter in the optimized kernel.
	if err = countTransforms(p.RingQ(), f.calls); err != nil {
		t.Fatal(err)
	}
	if err = countTransforms(p.RingP(), f.calls); err != nil {
		t.Fatal(err)
	}
	return f
}

func sameBytes(t *testing.T, a, b *rlwe.Ciphertext) {
	t.Helper()
	x, err := a.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	y, err := b.MarshalBinary()
	if err != nil || !bytes.Equal(x, y) {
		t.Fatalf("ciphertext or metadata differs byte-for-byte: %v", err)
	}
}

func checkCounters(t *testing.T, native, actual transformCounts, events []Event) {
	t.Helper()
	var sum transformCounts
	for _, ev := range events {
		sum.forward += ev.MeasuredLimbNTTsSaved
		sum.inverse += ev.MeasuredLimbINTTsSaved
		if ev.MeasuredSingleLimbNTTsSaved != ev.MeasuredLimbNTTsSaved+ev.MeasuredLimbINTTsSaved || ev.NetDecompNTTSaved != ev.MeasuredSingleLimbNTTsSaved {
			t.Fatalf("inconsistent signed observation: %+v", ev)
		}
	}
	if sum.forward != native.forward-actual.forward || sum.inverse != native.inverse-actual.inverse {
		t.Fatalf("events=%+v; independently counted native=%+v actual=%+v", events, native, actual)
	}
}

func TestEncryptedFixedBlocks(t *testing.T) {
	for alpha := 1; alpha <= 4; alpha++ {
		t.Run(fmt.Sprintf("alpha%d", alpha), func(t *testing.T) {
			level := 2
			if alpha > 2 {
				level = 5 // full block, shortened tail and a key built at higher level
			}
			f := encryptedFixture(t, alpha, level)
			var events []Event
			Observe = func(e Event) { events = append(events, e) }
			defer func() { Observe = nil }()
			for _, rot := range []bool{false, true} {
				for k := 0; k <= level; k++ {
					s := Site{ID: "test", Level: level, K: k}
					var got, dense, exact *rlwe.Ciphertext
					var err error
					*f.calls = transformCounts{}
					if rot {
						exact, err = f.e.RotateNew(f.rot, 1)
					} else {
						exact, err = f.e.RelinearizeNew(f.mul)
					}
					if err != nil {
						t.Fatal(err)
					}
					native := *f.calls
					events = nil
					*f.calls = transformCounts{}
					if rot {
						got, err = Giant(f.e, f.rot.CopyNew(), 1, s)
					} else {
						got, err = RelinearizeNew(f.e, f.mul, s)
					}
					if err != nil {
						t.Fatal(err)
					}
					checkCounters(t, native, *f.calls, events)
					if k == 0 {
						sameBytes(t, exact, got)
						if len(events) != 0 {
							t.Fatal("k=0 took the diagnostic/approximate path")
						}
						continue
					}
					rq := f.p.RingQ().AtLevel(level)
					if rot {
						ref := f.rot.CopyNew()
						ref.Value[1] = oracleNTT(rq, ref.Value[1], k)
						dense, err = f.e.RotateNew(ref, 1)
					} else {
						ref := f.mul.CopyNew()
						ref.Value[2] = oracleNTT(rq, ref.Value[2], k)
						dense, err = f.e.RelinearizeNew(ref)
					}
					if err != nil || len(events) != 1 {
						t.Fatalf("rot=%t k=%d events=%d err=%v", rot, k, len(events), err)
					}
					sameBytes(t, dense, got)
					ev := events[0]
					blocks := (level + alpha) / alpha
					if ev.Alpha != alpha || ev.LevelQ != level || ev.TruncatedLimbsK != k || ev.OriginalBlocks != blocks || ev.FullBlocksSkipped != k/alpha || ev.ActiveExternalProducts != blocks-k/alpha || ev.PartialBlockZeroLimbs != k%alpha || ev.Decompositions != ev.ActiveExternalProducts {
						t.Fatalf("rot=%t k=%d invalid physical work: %+v", rot, k, ev)
					}
					if alpha == 2 {
						v := make([]float64, len(f.truth))
						if err = f.encoder.Decode(f.dec.DecryptNew(got), v); err != nil {
							t.Fatal(err)
						}
						var peak float64
						for i, x := range v {
							j := i
							if rot {
								j = (i + 1) % len(v)
							}
							peak = math.Max(peak, math.Abs(x-f.truth[j]))
						}
						if !(peak < 1e-6) {
							t.Fatalf("decoded L_inf=%g", peak)
						}
						t.Logf("rotate=%t level=%d alpha=2 k=%d skipped=%d active=%d partial=%d NTT=%d->%d INTT=%d->%d signed_net=%d decoded_Linf=%.3g", rot, level, k, ev.FullBlocksSkipped, ev.ActiveExternalProducts, ev.PartialBlockZeroLimbs, ev.NativeForwardNTT, ev.ForwardNTT, ev.NativeInverseNTT, ev.InverseNTT, ev.MeasuredSingleLimbNTTsSaved, peak)
					}
				}
			}
		})
	}
}

func TestHoistedAccounting(t *testing.T) {
	f := encryptedFixture(t, 2, 2)
	var events []Event
	Observe = func(e Event) { events = append(events, e) }
	defer func() { Observe = nil }()
	shifts := []int{1, 2, 3}
	for _, ks := range [][]int{{0, 0, 0}, {1, 1, 1}, {1, 2, 1}, {1, 0, 2}} {
		sites := map[int]Site{}
		for i, k := range ks {
			sites[i+1] = Site{Level: 2, K: k}
		}
		*f.calls = transformCounts{}
		exact, err := f.e.RotateHoistedNew(f.rot, shifts)
		if err != nil {
			t.Fatal(err)
		}
		native := *f.calls
		*f.calls, events = transformCounts{}, nil
		got, err := Babies(f.e, f.rot, shifts, sites)
		if err != nil {
			t.Fatal(err)
		}
		checkCounters(t, native, *f.calls, events)
		seen := map[int]bool{}
		idx := 0
		for i, k := range ks {
			if k == 0 {
				sameBytes(t, got[i+1], exact[i+1])
				continue
			}
			ref := f.rot.CopyNew()
			ref.Value[1] = oracleNTT(f.p.RingQ().AtLevel(2), ref.Value[1], k)
			dense, err := f.e.RotateHoistedNew(ref, []int{i + 1})
			if err != nil {
				t.Fatal(err)
			}
			sameBytes(t, got[i+1], dense[i+1])
			if idx >= len(events) || (events[idx].Decompositions == 0) != seen[k] || events[idx].ActiveExternalProducts != 2-k/2 {
				t.Fatalf("ks=%v: bad shared charge: %+v", ks, events)
			}
			seen[k] = true
			idx++
		}
		if idx != len(events) {
			t.Fatalf("extraneous event: %+v", events)
		}
	}
}

func TestRejectShortPKey(t *testing.T) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 8, LogQ: []int{35, 40, 41}, LogP: []int{53, 54}, LogDefaultScale: 35})
	if err != nil {
		t.Fatal(err)
	}
	gen := rlwe.NewKeyGenerator(p)
	sk := gen.GenSecretKeyNew()
	zero := 0
	short := rlwe.EvaluationKeyParameters{LevelP: &zero}
	e := ckks.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(gen.GenRelinearizationKeyNew(sk, short), gen.GenGaloisKeyNew(p.GaloisElement(1), sk, short)))
	s := Site{ID: "short-P", Level: 2, K: 1}
	ct := ckks.NewCiphertext(p, 1, s.Level)
	before := ct.CopyNew()
	Observe = func(Event) { t.Fatal("work observed for an incompatible evaluation key") }
	defer func() { Observe = nil }()
	check := func(err error) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), "incompatible evaluation key levels") {
			t.Fatalf("expected concrete key-level rejection, got %v", err)
		}
	}
	check(Relinearize(e, ckks.NewCiphertext(p, 2, s.Level), ct, s))
	_, err = Giant(e, ct, 1, s)
	check(err)
	_, err = Babies(e, ct, []int{1}, map[int]Site{1: s})
	check(err)
	sameBytes(t, before, ct)
}
