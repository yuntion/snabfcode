// Package gadget implements fixed-key, pure-RNS centered-prefix key switching.
package gadget

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type Site struct {
	ID       string
	Level, K int
}

// CenteredGadgetStats is per external product invocation. Block counts describe
// the physical key. Signed transform savings are native minus actual, BEFORE
// the identical ModDown; a shared decomposition is charged only once per batch.
type CenteredGadgetStats struct {
	Alpha, LevelQ, TruncatedLimbsK                            int
	OriginalBlocks, FullBlocksSkipped, ActiveExternalProducts int
	PartialBlockZeroLimbs                                     int
	NativeForwardNTT, NativeInverseNTT                        int
	MeasuredLimbNTTsSaved, MeasuredLimbINTTsSaved             int
	MeasuredSingleLimbNTTsSaved                               int // forward + inverse
}

type Event struct {
	Site
	CenteredGadgetStats
	RowsExecuted, RowsSkipped, ForwardNTT, InverseNTT, Decompositions int
	ResidualMax, ResidualMean, ResidualMeanSquare                     float64
	LogScale                                                          float64
	NativeDecompNTT, NetDecompNTTSaved                                int
	ResidualVariance                                                  float64
}

// Observe is for serial diagnostics only. A non-nil observer opts in to a
// measured native reference; neither that replay nor events occur for k=0.
var Observe func(Event)

type hoist struct {
	rows []ringqp.Poly // indexed by ORIGINAL key block; nil Q storage means skipped
	info Event
}

func decompose(e *ckks.Evaluator, in ring.Poly, s Site) (*hoist, error) {
	p := e.GetParameters()
	alpha := p.PCount()
	if alpha < 1 || alpha > 4 || s.K < 1 || s.K > s.Level || s.Level > p.MaxLevelQ() || in.Level() < s.Level {
		return nil, fmt.Errorf("unsupported centered gadget site %+v (alpha=%d)", s, alpha)
	}
	qp := p.RingQP().AtLevel(s.Level, alpha-1)
	h := &hoist{rows: make([]ringqp.Poly, p.BaseRNSDecompositionVectorSize(s.Level, alpha-1))}
	h.info.Site, h.info.Alpha, h.info.LevelQ = s, alpha, s.Level
	a := project(qp.RingQ, in, s.K, &h.info)
	for b := range h.rows {
		lo, end := b*alpha, min((b+1)*alpha, s.Level+1)
		if end <= s.K {
			continue // no digit storage, basis extension, transforms or later MAC
		}
		row := qp.NewPoly()
		if lo < s.K {
			extendBoundary(qp, a, lo, s.K, end, row)
		} else {
			e.Decomposer.DecomposeAndSplit(s.Level, alpha-1, alpha, b, a, row.Q, row.P)
		}
		h.info.Decompositions++
		for i, sr := range qp.RingQ.SubRings[:s.Level+1] {
			src := row.Q.Coeffs[i]
			if lo <= i && i < end {
				if i < s.K {
					h.info.PartialBlockZeroLimbs++
					continue // native limb in THIS digit only; already zero in new row
				}
				src = a.Coeffs[i] // never reuse the unprojected input NTT
			}
			sr.NTT(src, row.Q.Coeffs[i])
			h.info.ForwardNTT++
		}
		for i, sr := range qp.RingP.SubRings[:alpha] {
			sr.NTT(row.P.Coeffs[i], row.P.Coeffs[i])
			h.info.ForwardNTT++
		}
		h.rows[b] = row
	}
	return h, nil
}

func checkKey(p *ckks.Parameters, key *rlwe.GadgetCiphertext, ct *rlwe.Ciphertext, s Site) error {
	alpha := p.PCount()
	if !ct.IsNTT || ct.IsMontgomery || alpha < 1 || alpha > 4 || s.Level > p.MaxLevelQ() {
		return fmt.Errorf("unsupported centered key-switch domain at %+v (alpha=%d, NTT=%t, Montgomery=%t)", s, alpha, ct.IsNTT, ct.IsMontgomery)
	}
	beta := p.BaseRNSDecompositionVectorSize(s.Level, alpha-1)
	if key.BaseTwoDecomposition != 0 || len(key.Value) < beta {
		return fmt.Errorf("incompatible pure-RNS evaluation key at %+v", s)
	}
	for _, row := range key.Value[:beta] {
		if len(row) != 1 || len(row[0]) != 2 {
			return fmt.Errorf("expected two expanded key polynomials per fixed block at %+v", s)
		}
		for _, poly := range row[0] {
			if poly.LevelP() != alpha-1 || poly.LevelQ() < s.Level || poly.Q.N() != p.N() || poly.P.N() != p.N() {
				return fmt.Errorf("incompatible evaluation key levels/ring at %+v", s)
			}
		}
	}
	return nil
}

func (h *hoist) product(e *ckks.Evaluator, key *rlwe.GadgetCiphertext, ct *rlwe.Ciphertext, s Site, first bool, native transformCounts) *rlwe.Ciphertext {
	l, lp := s.Level, h.info.Alpha-1
	qp := e.GetParameters().RingQP().AtLevel(l, lp)
	acc := rlwe.NewElementExtended(e.GetParameters(), 1, l, lp)
	*acc.MetaData = *ct.MetaData
	ev := h.info
	ev.Site, ev.LogScale = s, ct.LogScale()
	if !first {
		ev.ForwardNTT, ev.InverseNTT, ev.Decompositions = 0, 0, 0
	}
	for b, row := range h.rows {
		ev.OriginalBlocks++
		if row.Q.Coeffs == nil {
			ev.FullBlocksSkipped++
			continue
		}
		for j, keypoly := range key.Value[b][0] {
			if ev.ActiveExternalProducts == 0 {
				qp.MulCoeffsMontgomery(keypoly, row, acc.Value[j])
			} else {
				qp.MulCoeffsMontgomeryThenAdd(keypoly, row, acc.Value[j])
			}
		}
		ev.ActiveExternalProducts++
	}
	out := ckks.NewCiphertext(*e.GetParameters(), 1, l)
	*out.MetaData = *ct.MetaData
	e.ModDown(l, lp, acc, out)
	if Observe != nil {
		ev.RowsExecuted, ev.RowsSkipped = ev.ActiveExternalProducts, ev.FullBlocksSkipped
		ev.NativeForwardNTT, ev.NativeInverseNTT = native.forward, native.inverse
		ev.MeasuredLimbNTTsSaved = native.forward - ev.ForwardNTT
		ev.MeasuredLimbINTTsSaved = native.inverse - ev.InverseNTT
		ev.MeasuredSingleLimbNTTsSaved = ev.MeasuredLimbNTTsSaved + ev.MeasuredLimbINTTsSaved
		ev.NativeDecompNTT, ev.NetDecompNTTSaved = native.forward+native.inverse, ev.MeasuredSingleLimbNTTsSaved
		Observe(ev)
	}
	return out
}

func check(s Site, ct *rlwe.Ciphertext) error {
	if ct == nil || ct.MetaData == nil || ct.Level() != s.Level || s.K < 0 || s.K > s.Level {
		return fmt.Errorf("invalid operand or stale gadget site %+v", s)
	}
	return nil
}

func Relinearize(e *ckks.Evaluator, ct, out *rlwe.Ciphertext, s Site) error {
	if err := check(s, ct); err != nil {
		return err
	}
	if ct.Degree() != 2 || out == nil || out.Degree() < 1 || out.Level() < s.Level {
		return fmt.Errorf("invalid relin operand")
	}
	if s.K == 0 {
		return e.Relinearize(ct, out)
	}
	key, err := e.CheckAndGetRelinearizationKey()
	if err != nil {
		return err
	}
	if err = checkKey(e.GetParameters(), &key.GadgetCiphertext, ct, s); err != nil {
		return err
	}
	h, err := decompose(e, ct.Value[2], s)
	if err != nil {
		return err
	}
	var native transformCounts
	if Observe != nil {
		if native, err = measureNative(e.GetRLWEParameters(), s.Level, ct.Value[2], &key.GadgetCiphertext, false); err != nil {
			return err
		}
	}
	tmp := h.product(e, &key.GadgetCiphertext, ct, s, true, native)
	rq := e.GetParameters().RingQ().AtLevel(s.Level)
	rq.Add(ct.Value[0], tmp.Value[0], out.Value[0])
	rq.Add(ct.Value[1], tmp.Value[1], out.Value[1])
	out.Resize(1, s.Level)
	*out.MetaData = *ct.MetaData
	return nil
}

func RelinearizeNew(e *ckks.Evaluator, ct *rlwe.Ciphertext, s Site) (*rlwe.Ciphertext, error) {
	if err := check(s, ct); err != nil {
		return nil, err
	}
	if s.K == 0 {
		return e.RelinearizeNew(ct)
	}
	out := ckks.NewCiphertext(*e.GetParameters(), 1, ct.Level())
	return out, Relinearize(e, ct, out, s)
}

func rotations(e *ckks.Evaluator, ct *rlwe.Ciphertext, shifts []int, sites map[int]Site, shared bool) (map[int]*rlwe.Ciphertext, error) {
	nExact := 0
	for _, v := range shifts {
		s, ok := sites[v]
		if !ok {
			return nil, fmt.Errorf("missing rotation site %d", v)
		}
		if err := check(s, ct); err != nil {
			return nil, err
		}
		if ct.Degree() != 1 {
			return nil, fmt.Errorf("invalid rotation operand")
		}
		if s.K == 0 {
			nExact++
		} else if e.GetParameters().GaloisElement(v) == 1 {
			return nil, fmt.Errorf("cannot truncate an identity rotation: no key switch at %+v", s)
		}
	}
	if nExact > 0 && nExact == len(shifts) {
		return e.RotateHoistedNew(ct, shifts)
	}
	var out map[int]*rlwe.Ciphertext
	if nExact != 0 {
		exact := make([]int, 0, nExact)
		for _, v := range shifts {
			if sites[v].K == 0 {
				exact = append(exact, v)
			}
		}
		var err error
		if out, err = e.RotateHoistedNew(ct, exact); err != nil {
			return nil, err
		}
	} else {
		out = make(map[int]*rlwe.Ciphertext, len(shifts))
	}
	byK := make(map[int]*hoist)
	// One native hoist would serve ALL these shifts. If exact members are
	// present it was already paid in both arms; only extra work remains.
	credit := nExact == 0
	for _, v := range shifts {
		s := sites[v]
		if s.K == 0 {
			continue
		}
		gal := e.GetParameters().GaloisElement(v)
		key, err := e.CheckAndGetGaloisKey(gal)
		if err != nil {
			return nil, err
		}
		if err = checkKey(e.GetParameters(), &key.GadgetCiphertext, ct, s); err != nil {
			return nil, err
		}
		h := byK[s.K]
		first := h == nil
		if first {
			if h, err = decompose(e, ct.Value[1], s); err != nil {
				return nil, err
			}
			byK[s.K] = h
		}
		var native transformCounts
		if Observe != nil && (!shared || credit) {
			if native, err = measureNative(e.GetRLWEParameters(), s.Level, ct.Value[1], &key.GadgetCiphertext, shared); err != nil {
				return nil, err
			}
			credit = false
		}
		t := h.product(e, &key.GadgetCiphertext, ct, s, first, native)
		rq := e.GetParameters().RingQ().AtLevel(s.Level)
		rq.Add(t.Value[0], ct.Value[0], t.Value[0])
		dst := ckks.NewCiphertext(*e.GetParameters(), 1, s.Level)
		*dst.MetaData = *ct.MetaData
		for j := 0; j < 2; j++ {
			rq.AutomorphismNTTWithIndex(t.Value[j], e.AutomorphismIndex(gal), dst.Value[j])
		}
		out[v] = dst
	}
	return out, nil
}

func Babies(e *ckks.Evaluator, ct *rlwe.Ciphertext, v []int, s map[int]Site) (map[int]*rlwe.Ciphertext, error) {
	return rotations(e, ct, v, s, true)
}

func Giant(e *ckks.Evaluator, ct *rlwe.Ciphertext, v int, s Site) (*rlwe.Ciphertext, error) {
	if err := check(s, ct); err != nil {
		return nil, err
	}
	if s.K == 0 {
		return ct, e.Rotate(ct, v, ct)
	}
	r, err := rotations(e, ct, []int{v}, map[int]Site{v: s}, false)
	return r[v], err
}
