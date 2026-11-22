// Package epochs binds concrete parameters, keys and plaintexts for models
// emitted by the native target. Eval calls the generated Go function.
package epochs

import (
	"encoding/hex"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math"
	"math/bits"
	"slices"
	"snabf.local/compiler/src/backend"
)

type Epoch struct {
	N, Scale     int
	Q, P, Galois []uint64
	KeyID        string
	KeyBytes     int
	Relin        bool
}
type PlainBinding struct {
	Index, N, Level int
	LogScale        float64 // physical encoding scale, not the ciphertext result scale
	Values          []float64
}
type Program struct {
	Fingerprint                               string
	Source, Target, Width, Slots, GaloisBytes int
	Epochs                                    []Epoch
	Plain                                     []PlainBinding
	Ports                                     []ExactPort
	Switch                                    *Transition
}
type Evaluation func(*Runtime, *rlwe.Ciphertext) (*rlwe.Ciphertext, error)
type KeyRecord struct {
	Role                     string
	N, LevelQ, LevelP, Bytes int
	Galois                   uint64
	KeyID                    string
	TargetKeyID              string `json:",omitempty"`
}
type family struct {
	p    ckks.Parameters
	enc  *ckks.Encoder
	eval *ckks.Evaluator
}
type Runtime struct {
	Program  Program
	Keys     []KeyRecord
	families map[int]family
	encrypt  *rlwe.Encryptor
	decrypt  *rlwe.Decryptor
	convert  *backend.SparseDownConverter
	plain    map[int]*rlwe.Plaintext
	exact    map[string]rlwe.Scale
	evaluate Evaluation
	last     EvalObservation // per Eval; like the CKKS evaluator, this runtime is not concurrent

	diagonalRoots map[int]bool // only populated by successful guarded Bind
}
type PhysicalPort struct {
	N, Level int
	LogScale float64
}
type EvalObservation struct {
	Converted []string
	Join      []PhysicalPort
	JoinSite  string
}

func (r *Runtime) LastEval() EvalObservation {
	return EvalObservation{Converted: slices.Clone(r.last.Converted), Join: slices.Clone(r.last.Join), JoinSite: r.last.JoinSite}
}
func (r *Runtime) CheckJoin(site string, a, b *rlwe.Ciphertext) error {
	if a == nil || b == nil || a.MetaData == nil || b.MetaData == nil || len(a.Value) == 0 || len(b.Value) == 0 || a.Value[0].N() != r.Program.Target || b.Value[0].N() != r.Program.Target || a.Level() != b.Level() || !finite(a.LogScale()) || !finite(b.LogScale()) || math.Abs(a.LogScale()-b.LogScale()) > 1e-8 {
		return fmt.Errorf("unaligned physical terminal add: %s", site)
	}
	r.last.JoinSite = site
	r.last.Join = []PhysicalPort{{a.Value[0].N(), a.Level(), a.LogScale()}, {b.Value[0].N(), b.Level(), b.LogScale()}}
	return nil
}

type binaryKey interface{ MarshalBinary() ([]byte, error) }

func (r *Runtime) key(role string, n, l, p int, g uint64, k binaryKey) (int, error) {
	b, e := k.MarshalBinary()
	if e != nil {
		return 0, e
	}
	record := KeyRecord{Role: role, N: n, LevelQ: l, LevelP: p, Bytes: len(b), Galois: g}
	for _, e := range r.Program.Epochs {
		if e.N == n {
			record.KeyID = e.KeyID
			break
		}
	}
	if role == "down" {
		record.TargetKeyID = r.Program.Switch.ToKey
	}
	r.Keys = append(r.Keys, record)
	return len(b), nil
}

type EncodingPolicy uint8

const RequireExactEncoding EncodingPolicy = 1

// Two-argument Bind remains the archived engineering ABI. Newly emitted
// programs require receipts even when a caller removes the entire port list.
func Bind(s Program, evaluate Evaluation, policy ...EncodingPolicy) (*Runtime, error) {
	diagonals := len(policy) == 1 && policy[0] == RequireDiagonalEncodingV1
	exact := diagonals || (len(policy) == 1 && policy[0] == RequireExactEncoding)
	if !exact && (len(policy) != 0 || len(s.Ports) != 0) {
		return nil, fmt.Errorf("invalid encoding policy")
	}
	r := &Runtime{Program: s, evaluate: evaluate, families: map[int]family{}, plain: map[int]*rlwe.Plaintext{}}
	digest, err := hex.DecodeString(s.Fingerprint)
	if err != nil || len(digest) != 32 || evaluate == nil || s.Width < 1 || s.Slots < s.Width || s.Slots%s.Width != 0 || s.Slots&(s.Slots-1) != 0 {
		return nil, fmt.Errorf("invalid epoch ABI")
	}
	if err := validateDomains(s); err != nil {
		return nil, err
	}
	var scales map[int]rlwe.Scale
	if exact {
		r.exact, scales, err = validateExactPorts(s)
		if err != nil {
			return nil, err
		}
	}
	sk := map[int]*rlwe.SecretKey{}
	total := 0
	for _, e := range s.Epochs {
		p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: bits.TrailingZeros(uint(e.N)), Q: e.Q, P: e.P, LogDefaultScale: e.Scale})
		if err != nil {
			return nil, err
		}
		if p.N() != e.N || p.MaxSlots() < s.Slots || !slices.Equal(p.Q(), e.Q) || !slices.Equal(p.P(), e.P) || sk[e.N] != nil {
			return nil, fmt.Errorf("invalid epoch parameters")
		}
		g := rlwe.NewKeyGenerator(p)
		sk[e.N] = g.GenSecretKeyNew()
		var rel *rlwe.RelinearizationKey
		if e.N == s.Source {
			pk := g.GenPublicKeyNew(sk[e.N])
			r.encrypt = rlwe.NewEncryptor(p, pk)
			if _, err = r.key("public", e.N, p.MaxLevel(), p.MaxLevelP(), 0, pk); err != nil {
				return nil, err
			}
		}
		if e.N == s.Target {
			r.decrypt = rlwe.NewDecryptor(p, sk[e.N])
		}
		if e.Relin {
			rel = g.GenRelinearizationKeyNew(sk[e.N])
			if _, err = r.key("relin", e.N, rel.LevelQ(), rel.LevelP(), 0, rel); err != nil {
				return nil, err
			}
		}
		var gs []*rlwe.GaloisKey
		for _, v := range e.Galois {
			k := g.GenGaloisKeyNew(v, sk[e.N])
			size, err := r.key("galois", e.N, k.LevelQ(), k.LevelP(), v, k)
			if err != nil {
				return nil, err
			}
			if size != e.KeyBytes {
				return nil, fmt.Errorf("serialized Galois key size mismatch")
			}
			total += size
			gs = append(gs, k)
		}
		r.families[e.N] = family{p, ckks.NewEncoder(p), ckks.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(rel, gs...))}
	}
	if total != s.GaloisBytes || r.encrypt == nil || r.decrypt == nil {
		return nil, fmt.Errorf("stale physical key inventory")
	}
	if s.Target != s.Source {
		large, small := r.families[s.Source].p, r.families[s.Target].p
		l, lp := s.Switch.LevelQ, s.Switch.LevelP
		if l != small.MaxLevel() || lp != small.MaxLevelP() {
			return nil, fmt.Errorf("wrong boundary modulus")
		}
		k := rlwe.NewKeyGenerator(large).GenEvaluationKeyNew(sk[s.Source], sk[s.Target], rlwe.EvaluationKeyParameters{LevelQ: &l, LevelP: &lp})
		size, err := r.key("down", large.N(), k.LevelQ(), k.LevelP(), 0, k)
		if err != nil {
			return nil, err
		}
		if k.LevelQ() != l || k.LevelP() != lp || size != s.Switch.KeyBytes {
			return nil, fmt.Errorf("conversion key not at screened QP")
		}
		r.convert, err = backend.NewSparseDownConverter(large, small, k)
		if err != nil {
			return nil, err
		}
	}
	if diagonals {
		r.diagonalRoots = make(map[int]bool)
	}
	for _, i := range s.Plain {
		f, ok := r.families[i.N]
		if !ok || i.Level < 0 || i.Level > f.p.MaxLevel() || i.Index < 0 || r.plain[i.Index] != nil {
			return nil, fmt.Errorf("invalid plaintext binding %d", i.Index)
		}
		pt := ckks.NewPlaintext(f.p, i.Level)
		pt.LogDimensions.Cols = bits.TrailingZeros(uint(s.Slots))
		if exact {
			pt.Scale = scales[i.Index]
			if err := checkEncoderPort(s.Ports[i.Index], pt, f.enc); err != nil {
				return nil, err
			}
			if diagonals && !r.diagonalRoots[i.N] && f.enc.Prec() == 53 &&
				i.N <= 65536 && s.Slots >= 16 && s.Slots <= 32768 {
				// NewEncoder uses this same deterministic v6.2.0 table constructor.
				m := int(f.p.RingQ().NthRoot())
				if m != 2*i.N || !checkDiagonalRoots(ckks.GetRootsComplex128(m), m, min(64, s.Slots)) {
					return nil, fmt.Errorf("unproved periodic diagonal roots: %s", s.Ports[i.Index].ID)
				}
				r.diagonalRoots[i.N] = true
			}
		} else {
			scale := math.Exp2(i.LogScale) // legacy generated engineering programs only
			if !finite(scale) || scale <= 0 {
				return nil, fmt.Errorf("invalid encoding scale at %d", i.Index)
			}
			pt.Scale = rlwe.NewScale(scale)
		}
		if len(i.Values) == 0 || s.Slots%len(i.Values) != 0 {
			return nil, fmt.Errorf("invalid constants at %d", i.Index)
		}
		v := make([]float64, s.Slots)
		for k := range v {
			v[k] = i.Values[k%len(i.Values)]
		}
		if err := f.enc.Encode(v, pt); err != nil {
			return nil, err
		}
		r.plain[i.Index] = pt
	}
	return r, nil
}
func (r *Runtime) Encrypt(x []float64) (*rlwe.Ciphertext, error) {
	s := r.Program
	if len(x) != s.Width {
		return nil, fmt.Errorf("wrong logical width")
	}
	f := r.families[s.Source]
	pt := ckks.NewPlaintext(f.p, f.p.MaxLevel())
	pt.LogDimensions.Cols = bits.TrailingZeros(uint(s.Slots))
	if r.exact != nil {
		if err := checkEncoderPort(s.Ports[0], pt, f.enc); err != nil {
			return nil, err
		}
	}
	v := make([]float64, s.Slots)
	for j := range v {
		v[j] = x[j%s.Width]
	}
	if err := f.enc.Encode(v, pt); err != nil {
		return nil, err
	}
	return r.encrypt.EncryptNew(pt)
}
func (r *Runtime) Eval(input *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	r.last = EvalObservation{}
	v, err := r.evaluate(r, input)
	if err == nil && r.Program.Switch != nil && len(r.last.Converted) != len(r.Program.Switch.Operands) {
		return nil, fmt.Errorf("unconverted live ciphertext")
	}
	return v, err
}

func (r *Runtime) Evaluator(n int) *ckks.Evaluator { return r.families[n].eval }
func (r *Runtime) Plain(i int) *rlwe.Plaintext     { return r.plain[i] }

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func (r *Runtime) Check(id string, v *rlwe.Ciphertext, n, l int, scale float64) error {
	if v == nil || v.MetaData == nil || len(v.Value) == 0 || v.Value[0].N() != n || v.Level() != l || !v.IsBatched || v.LogDimensions.Rows != 0 || v.Slots() != r.Program.Slots || !v.IsNTT || !finite(v.LogScale()) || math.Abs(v.LogScale()-scale) > 1e-7 {
		return fmt.Errorf("physical epoch/level/scale/layout mismatch: %s", id)
	}
	if r.exact != nil {
		want, ok := r.exact[id]
		if !ok || v.Scale.Mod != nil || v.Scale.Value.Prec() != rlwe.ScalePrecision || !want.Equal(v.Scale) {
			return fmt.Errorf("missing or mismatched exact physical scale: %s", id)
		}
	}
	return nil
}

func (r *Runtime) Down(edge string, v *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	t := r.Program.Switch
	if r.convert == nil || t == nil || slices.Contains(r.last.Converted, edge) {
		return nil, fmt.Errorf("unbound conversion")
	}
	for _, op := range t.Operands {
		if edge != op.Edge {
			continue
		}
		if err := r.Check(edge, v, op.Provided.N, t.LevelQ, op.Provided.LogScale); err != nil {
			return nil, err
		}
		out, err := r.convert.Convert(v)
		if err == nil {
			r.last.Converted = append(r.last.Converted, edge)
		}
		return out, err
	}
	return nil, fmt.Errorf("unbound conversion: %s", edge)
}
func (r *Runtime) Decode(ct *rlwe.Ciphertext) ([]float64, error) {
	if ct == nil || ct.MetaData == nil || len(ct.Value) == 0 || ct.Value[0].N() != r.Program.Target || ct.Slots() != r.Program.Slots {
		return nil, fmt.Errorf("wrong result key domain")
	}
	v := make([]float64, r.Program.Slots)
	return v, r.families[r.Program.Target].enc.Decode(r.decrypt.DecryptNew(ct), v)
}
