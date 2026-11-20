package ringswitch

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"slices"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// FullSlotSplitter is a standalone, serial evaluation primitive for a single
// standard-ring 2:1 cut. It is not a layout, security or emission receipt.
type FullSlotSplitter struct {
	src, dst        ckks.Parameters
	level, used     int
	eval            *ckks.Evaluator
	down            *rlwe.EvaluationKey
	masks           [2]*rlwe.Plaintext
	gal             uint64
	slotMap, downID string
	downBytes       int
}

type SplitOutput struct {
	N             int     `json:"n"`
	Level         int     `json:"level"`
	LogicalStart  int     `json:"logical_start"`
	UsedSlots     int     `json:"used_slots"`
	LogScale      float64 `json:"log2_scale"`
	DownKeySHA256 string  `json:"down_key_sha256"`
}
type SplitObservation struct {
	MaskMulCount      int            `json:"mask_mul_count"`
	RescaleCount      int            `json:"rescale_count"`
	FoldRotCount      int            `json:"fold_rot_count"`
	DownConvCount     int            `json:"downconv_count"`
	SlotMapID         string         `json:"slot_map_id"`
	FoldGaloisElement uint64         `json:"fold_galois_element"`
	SwitchKeyBytes    int            `json:"switch_key_bytes"`
	Outputs           [2]SplitOutput `json:"outputs"`
}

// Restrict each source embedding to Y=X^2. Only the natural full-capacity
// Lattigo standard layout is supported. Derive both representatives using
// the parameters' Galois maps; do not infer slot order from input values.
func splitSlotMap(src, dst ckks.Parameters) (pairs [][2]int, id string, err error) {
	s := dst.MaxSlots()
	mod := uint64(2 * dst.N())
	root := uint64(2 * src.N())
	half := src.GaloisElement(s)
	target := make(map[uint64]int, s)
	for j := range s {
		target[dst.GaloisElement(j)] = j
	}
	if len(target) != s {
		return nil, "", fmt.Errorf("nonbijective target slot map")
	}
	pairs = make([][2]int, s)
	counts := make([]int, s)
	for i := range src.MaxSlots() {
		j, ok := target[src.GaloisElement(i)%mod]
		if !ok || counts[j] >= 2 {
			return nil, "", fmt.Errorf("unrecognized CKKS trace orbit")
		}
		pairs[j][counts[j]] = i
		counts[j]++
	}
	h := sha256.New()
	h.Write([]byte("snabf.standard-ckks-half-slots.v1\x00"))
	put := func(x uint64) { var b [8]byte; binary.LittleEndian.PutUint64(b[:], x); h.Write(b[:]) }
	put(root)
	put(mod)
	put(half)
	for j, ab := range pairs {
		if counts[j] != 2 || ab[0] != j || ab[1] != j+s || (src.GaloisElement(ab[0])*half)%root != src.GaloisElement(ab[1]) {
			return nil, "", fmt.Errorf("unsupported slot order or fold automorphism at %d", j)
		}
		put(uint64(ab[0]))
		put(uint64(ab[1]))
		put(dst.GaloisElement(j))
	}
	return pairs, fmt.Sprintf("%x", h.Sum(nil)), nil
}

// Check the expanded large-ring key BEFORE handing any rows to Lattigo.
func splitKey(p ckks.Parameters, key *rlwe.EvaluationKey, level int) bool {
	if key == nil || key.BaseTwoDecomposition != 0 || len(key.Value) == 0 || len(key.Value[0]) != 1 || len(key.Value[0][0]) != 2 {
		return false
	}
	l := key.Value[0][0][0].LevelQ()
	lp := p.MaxLevelP()
	if l < level || l > p.MaxLevel() || len(key.Value) != p.BaseRNSDecompositionVectorSize(l, lp) {
		return false
	}
	for _, row := range key.Value {
		if len(row) != 1 || len(row[0]) != 2 {
			return false
		}
		for _, poly := range row[0] {
			if poly.LevelQ() != l || poly.LevelP() != lp {
				return false
			}
			for _, r := range []ring.Poly{poly.Q, poly.P} {
				for _, coeffs := range r.Coeffs {
					if len(coeffs) != p.N() {
						return false
					}
				}
			}
		}
	}
	return true
}

// The caller owns domain binding: fold is the source half-period Galois key
// and down must switch the source secret to the embedded destination secret.
// Key bytes identify reuse, not the secret relationship or estimator result.
func NewFullSlotSplitter(src, dst ckks.Parameters, level, usedSlots int, fold *rlwe.GaloisKey, down *rlwe.EvaluationKey) (*FullSlotSplitter, error) {
	if src.RingType() != ring.Standard || dst.RingType() != ring.Standard || src.N() != 2*dst.N() || usedSlots <= dst.MaxSlots() || usedSlots > src.MaxSlots() {
		return nil, fmt.Errorf("full-slot split requires N=2N' and N'/2 < used_slots <= N/2")
	}
	if level < 1 || level > src.MaxLevel() || level-1 > dst.MaxLevel() || src.LevelsConsumedPerRescaling() != 1 || dst.QCount() > src.QCount() || !slices.Equal(src.Q()[:dst.QCount()], dst.Q()) || src.PCount() < 1 || src.PCount() > 4 || !slices.Equal(src.P(), dst.P()) {
		return nil, fmt.Errorf("incompatible split level or ordered Q/P basis")
	}
	gal := src.GaloisElement(dst.MaxSlots())
	if fold == nil || fold.GaloisElement != gal || fold.NthRoot != src.RingQ().NthRoot() || !splitKey(src, &fold.EvaluationKey, level-1) || !splitKey(src, down, level-1) {
		return nil, fmt.Errorf("incompatible half-period rotation or conversion key")
	}
	pairs, id, err := splitSlotMap(src, dst)
	if err != nil {
		return nil, err
	}
	down = down.CopyNew()
	raw, err := down.MarshalBinary()
	if err != nil {
		return nil, err
	}
	f := &FullSlotSplitter{src: src, dst: dst, level: level, used: usedSlots, eval: ckks.NewEvaluator(src, rlwe.NewMemEvaluationKeySet(nil, fold.CopyNew())), down: down, slotMap: id, gal: gal, downID: fmt.Sprintf("%x", sha256.Sum256(raw)), downBytes: len(raw)}
	encoder := ckks.NewEncoder(src)
	for half := range 2 {
		mask := make([]float64, src.MaxSlots())
		for _, ab := range pairs {
			mask[ab[half]] = 1
		}
		pt := ckks.NewPlaintext(src, level)
		pt.Scale = rlwe.NewScale(src.Q()[level])
		if err = encoder.Encode(mask, pt); err != nil {
			return nil, err
		}
		f.masks[half] = pt
	}
	return f, nil
}

func (f *FullSlotSplitter) fullOperand(ct *rlwe.Ciphertext, level int) bool {
	if ct == nil || ct.MetaData == nil || ct.Degree() != 1 || !ct.IsBatched || !ct.IsNTT || ct.IsMontgomery || ct.IsBitReversed || ct.LogDimensions != f.src.LogMaxDimensions() {
		return false
	}
	if ct.Scale.Mod != nil || ct.Scale.Value.Sign() <= 0 || ct.Scale.Value.IsInf() {
		return false
	}
	for _, poly := range ct.Value {
		if poly.Level() != level {
			return false
		}
		for _, row := range poly.Coeffs {
			if len(row) != f.src.N() {
				return false
			}
		}
	}
	return true
}

type foldedHalf struct {
	owner *FullSlotSplitter
	half  int
	ct    *rlwe.Ciphertext
}

func (f *FullSlotSplitter) fold(in *rlwe.Ciphertext, half int, r *SplitObservation) (foldedHalf, error) {
	t, e := f.eval.MulNew(in, f.masks[half])
	if e != nil {
		return foldedHalf{}, e
	}
	r.MaskMulCount++
	if e = f.eval.Rescale(t, t); e != nil {
		return foldedHalf{}, e
	}
	r.RescaleCount++
	g, e := f.eval.RotateNew(t, f.dst.MaxSlots())
	if e != nil {
		return foldedHalf{}, e
	}
	r.FoldRotCount++
	p, e := f.eval.AddNew(t, g)
	if e != nil {
		return foldedHalf{}, e
	}
	return foldedHalf{f, half, p}, nil
}

// Only fold constructs this private input. The full source metadata is
// retained during the physical normalized trace. ApplyEvaluationKey copies
// it to its output; assign the proven target layout only AFTER projection.
func (f *FullSlotSplitter) projectFolded(v foldedHalf) (*rlwe.Ciphertext, error) {
	if v.owner != f || v.half < 0 || v.half > 1 || !f.fullOperand(v.ct, f.level-1) {
		return nil, fmt.Errorf("missing checked fold producer")
	}
	y := ckks.NewCiphertext(f.dst, 1, f.level-1)
	if e := f.eval.ApplyEvaluationKey(v.ct, f.down, y); e != nil {
		return nil, e
	}
	y.LogDimensions = f.dst.LogMaxDimensions()
	return y, nil
}

// Split returns two real target-domain ciphertexts; no decryptor is used.
// Output i represents the logical interval beginning at i*(N'/2).
func (f *FullSlotSplitter) Split(in *rlwe.Ciphertext) ([2]*rlwe.Ciphertext, SplitObservation, error) {
	var out [2]*rlwe.Ciphertext
	var r SplitObservation
	if !f.fullOperand(in, f.level) {
		return out, r, fmt.Errorf("expected bound full-capacity source NTT ciphertext at level %d", f.level)
	}
	r.SlotMapID, r.FoldGaloisElement, r.SwitchKeyBytes = f.slotMap, f.gal, f.downBytes
	for i := range out {
		v, e := f.fold(in, i, &r)
		if e != nil {
			return [2]*rlwe.Ciphertext{}, r, e
		}
		out[i], e = f.projectFolded(v)
		if e != nil {
			return [2]*rlwe.Ciphertext{}, r, e
		}
		r.DownConvCount++
		n := f.dst.MaxSlots()
		r.Outputs[i] = SplitOutput{N: out[i].Value[0].N(), Level: out[i].Level(), LogicalStart: i * n, UsedSlots: min(n, f.used-i*n), LogScale: out[i].LogScale(), DownKeySHA256: f.downID}
	}
	return out, r, nil
}
