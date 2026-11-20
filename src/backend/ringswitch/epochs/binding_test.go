package epochs

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/ringswitch/epochs/heirckks"
)

func TestSourceEvaluationSharesKeysAndPublicEncoder(t *testing.T) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 10, LogQ: []int{45, 35}, LogP: []int{44}, LogDefaultScale: 35})
	if err != nil {
		t.Fatal(err)
	}
	s := Program{Source: p.N(), Target: p.N(), Width: 2, Slots: 8, Fingerprint: strings.Repeat("ab", 32), Epochs: []Epoch{{N: p.N(), Scale: 35, Q: p.Q(), P: p.P(), KeyID: "test/source"}}}
	r, err := Bind(s, func(r *Runtime, x *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
		return heirckks.Add(r.Evaluator(p.N()), x, x), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	e, q, enc := r.SourceEvaluation()
	if e != r.families[p.N()].eval || enc != r.families[p.N()].enc || !p.Equal(&q) {
		t.Fatal("source evaluation context was recreated")
	}
	x, err := r.Encrypt([]float64{.25, -.375})
	if err != nil {
		t.Fatal(err)
	}
	a, err := r.Eval(x.CopyNew())
	if err != nil {
		t.Fatal(err)
	}
	b := heirckks.Add(e, x, x)
	if !a.Equal(b) {
		t.Fatal("same ciphertext evaluated with two contexts gave different ciphertexts")
	}
	v, err := r.Decode(b)
	if err != nil {
		t.Fatal(err)
	}
	for j, y := range v {
		if math.Abs(y-[]float64{.5, -.75}[j%2]) > 1./4096 {
			t.Fatal(j, y)
		}
	}
}

func TestRejectUnboundSwitchBeforeKeyGeneration(t *testing.T) {
	const large, small = 1024, 512
	l := Layout{N: large, Capacity: large / 2, Width: 2, Period: 8, Stride: 1, Level: 1, Lanes: 1, Packing: "periodic_single", LogScale: 35, Offsets: []int{0}, Active: []int{0}}
	target := l
	target.N, target.Capacity = small, small/2
	a := Epoch{N: large, Scale: 35, Q: []uint64{17, 19, 23}, P: []uint64{31}, KeyID: "a"}
	b := Epoch{N: small, Scale: 35, Q: a.Q[:2], P: a.P, KeyID: "b"}
	s := Program{Source: large, Target: small, Width: 2, Slots: 8, Epochs: []Epoch{a, b}}
	tr := Transition{FromKey: "a", ToKey: "b", LevelQ: 1, LevelP: 0, KeyBytes: 1, Parallel: 1, Provided: l, Required: target,
		Cut: CutOperand{SSA: "op1.r0", Edge: "poly.n3", KeyID: "a", User: "snabf.convert"}, Basis: SwitchBasis{Ring: large, EmbeddedRing: small, Q: b.Q, P: b.P}}
	s.Switch = &tr
	if err := validateDomains(s); err != nil {
		t.Fatal(err)
	}
	for name, alter := range map[string]func(*Transition){
		"domain": func(x *Transition) { x.ToKey = x.FromKey }, "level": func(x *Transition) { x.LevelQ++ },
		"scale": func(x *Transition) { x.Required.LogScale++ }, "lane": func(x *Transition) { x.Provided.Active = []int{0, 1} },
		"storage": func(x *Transition) { x.Cut.User = "tensor.insert" }, "crossing": func(x *Transition) { x.Cut.KeyID = "b" },
		"basis": func(x *Transition) { x.Basis.Q = a.Q }, "ring": func(x *Transition) { x.Basis.EmbeddedRing = large },
	} {
		t.Run(name, func(t *testing.T) {
			x := tr
			alter(&x)
			s.Switch = &x
			if validateDomains(s) == nil {
				t.Fatal("accepted stale transition")
			}
		})
	}
}

func TestHEIRErrorIdentifiesSite(t *testing.T) {
	cause := errors.New("bad CKKS operand")
	wrapped := func() (err error) { site := "poly.n2.mul"; defer HEIRError(&site, &err); panic(cause) }()
	if !errors.Is(wrapped, cause) || !strings.Contains(wrapped.Error(), "poly.n2.mul") {
		t.Fatal(wrapped)
	}
}
