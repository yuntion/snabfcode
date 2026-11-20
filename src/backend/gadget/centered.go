// Package gadget implements the one-P, pure-RNS centered-prefix gadget.
package gadget

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/ring/ringqp"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math"
	"math/big"
)

type Site struct {
	ID       string
	Level, K int
}
type Event struct {
	Site
	RowsExecuted, RowsSkipped, ForwardNTT, InverseNTT, Decompositions int
	ResidualMax, ResidualMean, ResidualMeanSquare                     float64
	LogScale                                                          float64
	NativeDecompNTT, NetDecompNTTSaved                                int
	ResidualVariance                                                  float64
}

// An observer is installed only around a serial diagnostic inference, never in timed trials.
var Observe func(Event)

func centered(x, q uint64) int64 {
	if x > q/2 {
		return int64(x) - int64(q)
	}
	return int64(x)
}
func subCentered(x uint64, r int64, q uint64) uint64 {
	// q0 need not be smaller than the retained qi.
	v := r % int64(q)
	if v < 0 {
		v += int64(q)
	}
	x += q - uint64(v)
	if x >= q {
		x -= q
	}
	return x
}

// x' = x - center_D(x), D=product(q[:k]). The residual is ONE integer
// per coefficient, not k independently centered CRT digits. All first k
// components of x' are identically zero. The remaining rows keep modulus Q.
func project(rq *ring.Ring, in ring.Poly, k int, stats *Event) ring.Poly {
	a := rq.NewPoly()
	rq.INTT(in, a)
	stats.InverseNTT += rq.Level() + 1
	q := rq.ModuliChain()
	var mx, sum, ss float64
	note := func(r float64) {
		if Observe != nil {
			mx = math.Max(mx, math.Abs(r))
			sum += r
			ss += r * r
		}
	}
	if k == 1 {
		for j, x := range a.Coeffs[0] {
			r := centered(x, q[0])
			note(float64(r))
			for i := 1; i <= rq.Level(); i++ {
				a.Coeffs[i][j] = subCentered(a.Coeffs[i][j], r, q[i])
			}
		}
	} else {
		z := make([]*big.Int, rq.N())
		for j := range z {
			z[j] = new(big.Int)
		}
		rq.AtLevel(k-1).PolyToBigintCentered(a, 1, z)
		// Lattigo v6.2 centers floor(D/2) on the negative side; restore the
		// unique nearest representative for the odd-modulus boundary as well.
		D := rq.AtLevel(k - 1).ModulusAtLevel[k-1]
		half := new(big.Int).Rsh(new(big.Int).Set(D), 1)
		negative := new(big.Int).Neg(half)
		for _, r := range z {
			if r.Cmp(negative) < 0 {
				r.Add(r, D)
			}
		}
		var qi, tmp big.Int
		for j, r := range z {
			f, _ := r.Float64()
			note(f)
			for i := k; i <= rq.Level(); i++ {
				qi.SetUint64(q[i])
				u := tmp.Mod(r, &qi).Uint64()
				a.Coeffs[i][j] = subCentered(a.Coeffs[i][j], int64(u), q[i])
			}
		}
	}
	for i := 0; i < k; i++ {
		clear(a.Coeffs[i])
	}
	n := float64(rq.N())
	stats.ResidualMax = mx
	stats.ResidualMean = sum / n
	stats.ResidualMeanSquare = ss / n
	stats.ResidualVariance = math.Max(0, ss/n-(sum/n)*(sum/n))
	return a
}

// One hoist can feed several automorphisms. Transform every surviving row
// once, including its native limb (whose input NTT changed under projection).
type hoist struct {
	rows []ringqp.Poly
	info Event
}

func decompose(e *ckks.Evaluator, in ring.Poly, s Site) (*hoist, error) {
	p := e.GetParameters()
	if p.PCount() != 1 || s.K < 1 || s.K > s.Level || in.Level() < s.Level {
		return nil, fmt.Errorf("unsupported centered gadget site %+v", s)
	}
	rq := p.RingQ().AtLevel(s.Level)
	rqp := p.RingQP().AtLevel(s.Level, 0)
	h := &hoist{info: Event{Site: s}}
	a := project(rq, in, s.K, &h.info)
	for i := s.K; i <= s.Level; i++ {
		row := rqp.NewPoly()
		e.Decomposer.DecomposeAndSplit(s.Level, 0, 1, i, a, row.Q, row.P)
		rqp.NTT(row, row)
		h.rows = append(h.rows, row)
		h.info.ForwardNTT += s.Level + 2
		h.info.Decompositions++
	}
	return h, nil
}
func (h *hoist) product(e *ckks.Evaluator, key *rlwe.GadgetCiphertext, ct *rlwe.Ciphertext, site Site, first bool, nativeNTT int) (*rlwe.Ciphertext, error) {
	l := site.Level
	if key.LevelP() != 0 || key.BaseTwoDecomposition != 0 || key.LevelQ() < l || !ct.IsNTT || ct.IsMontgomery {
		return nil, fmt.Errorf("incompatible evaluation key/domain")
	}
	qp := e.GetParameters().RingQP().AtLevel(l, 0)
	acc := rlwe.NewElementExtended(e.GetParameters(), 1, l, 0)
	*acc.MetaData = *ct.MetaData
	ev := h.info
	ev.Site = site
	ev.LogScale = ct.LogScale()
	if !first {
		ev.ForwardNTT = 0
		ev.InverseNTT = 0
		ev.Decompositions = 0
	}
	ev.RowsSkipped = site.K
	ev.NativeDecompNTT = nativeNTT
	ev.NetDecompNTTSaved = nativeNTT - ev.ForwardNTT - ev.InverseNTT
	for i, row := range h.rows {
		keyrow := key.Value[site.K+i][0]
		for j := 0; j < 2; j++ {
			if i == 0 {
				qp.MulCoeffsMontgomery(keyrow[j], row, acc.Value[j])
			} else {
				qp.MulCoeffsMontgomeryThenAdd(keyrow[j], row, acc.Value[j])
			}
		}
		ev.RowsExecuted++
	}
	out := ckks.NewCiphertext(*e.GetParameters(), 1, l)
	*out.MetaData = *ct.MetaData
	e.ModDown(l, 0, acc, out)
	if Observe != nil {
		Observe(ev)
	}
	return out, nil
}
func check(s Site, ct *rlwe.Ciphertext) error {
	if ct.Level() != s.Level || s.K < 0 || s.K > s.Level {
		return fmt.Errorf("stale gadget site %+v at L%d", s, ct.Level())
	}
	return nil
}
func Relinearize(e *ckks.Evaluator, ct, out *rlwe.Ciphertext, s Site) error {
	if err := check(s, ct); err != nil {
		return err
	}
	if ct.Degree() != 2 || out.Level() < s.Level {
		return fmt.Errorf("invalid relin operand")
	}
	if s.K == 0 {
		return e.Relinearize(ct, out)
	}
	key, err := e.GetRelinearizationKey()
	if err != nil {
		return err
	}
	h, err := decompose(e, ct.Value[2], s)
	if err != nil {
		return err
	}
	tmp, err := h.product(e, &key.GadgetCiphertext, ct, s, true, (s.Level+1)*(s.Level+3))
	if err != nil {
		return err
	}
	rq := e.GetParameters().RingQ().AtLevel(s.Level)
	rq.Add(ct.Value[0], tmp.Value[0], out.Value[0])
	rq.Add(ct.Value[1], tmp.Value[1], out.Value[1])
	out.Resize(1, s.Level)
	*out.MetaData = *ct.MetaData
	return nil
}
func RelinearizeNew(e *ckks.Evaluator, ct *rlwe.Ciphertext, s Site) (*rlwe.Ciphertext, error) {
	out := ckks.NewCiphertext(*e.GetParameters(), 1, ct.Level())
	return out, Relinearize(e, ct, out, s)
}

func rotations(e *ckks.Evaluator, ct *rlwe.Ciphertext, shifts []int, sites map[int]Site, shared bool) (map[int]*rlwe.Ciphertext, error) {
	out := map[int]*rlwe.Ciphertext{}
	byK := map[int]*hoist{}
	var exact []int
	credit := true
	for _, v := range shifts {
		if sites[v].K == 0 {
			credit = false
		}
	}
	for _, v := range shifts {
		s, ok := sites[v]
		if !ok {
			return nil, fmt.Errorf("missing rotation site %d", v)
		}
		if err := check(s, ct); err != nil {
			return nil, err
		}
		if s.K == 0 {
			exact = append(exact, v)
			continue
		}
		h := byK[s.K]
		first := h == nil
		if first {
			var err error
			h, err = decompose(e, ct.Value[1], s)
			if err != nil {
				return nil, err
			}
			byK[s.K] = h
		}
		gal := e.GetParameters().GaloisElement(v)
		key, err := e.GetGaloisKey(gal)
		if err != nil {
			return nil, err
		}
		native := 0
		if !shared {
			native = (s.Level + 1) * (s.Level + 3)
		} else if credit {
			native = (s.Level + 1) * (s.Level + 2)
			credit = false
		}
		t, err := h.product(e, &key.GadgetCiphertext, ct, s, first, native)
		if err != nil {
			return nil, err
		}
		rq := e.GetParameters().RingQ().AtLevel(s.Level)
		rq.Add(t.Value[0], ct.Value[0], t.Value[0])
		dst := ckks.NewCiphertext(*e.GetParameters(), 1, s.Level)
		*dst.MetaData = *ct.MetaData
		for j := 0; j < 2; j++ {
			rq.AutomorphismNTTWithIndex(t.Value[j], e.AutomorphismIndex(gal), dst.Value[j])
		}
		out[v] = dst
	}
	if len(exact) > 0 {
		if shared {
			r, err := e.RotateHoistedNew(ct, exact)
			if err != nil {
				return nil, err
			}
			for k, v := range r {
				out[k] = v
			}
		} else {
			for _, v := range exact {
				r, err := e.RotateNew(ct, v)
				if err != nil {
					return nil, err
				}
				out[v] = r
			}
		}
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
