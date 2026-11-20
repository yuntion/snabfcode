package gadget

import (
	"fmt"
	"math/big"
	"slices"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// CalibrationCall describes one invocation in DirectGo, not one consumer of
// a shared decomposition. Shift and K order is the emitted order.
type CalibrationCall struct {
	Op        string
	Level     int
	Shifts, K []int
}

func (c CalibrationCall) Validate(n, levels int) error {
	width := len(c.Shifts)
	switch c.Op {
	case "relin":
		if width != 0 {
			return fmt.Errorf("relin has no shifts")
		}
		width = 1
	case "giant":
		if width != 1 {
			return fmt.Errorf("giant requires one shift")
		}
	case "baby":
		if width == 0 {
			return fmt.Errorf("empty baby group")
		}
	default:
		return fmt.Errorf("unsupported whole site %q", c.Op)
	}
	if c.Level < 0 || c.Level >= levels || len(c.K) != width {
		return fmt.Errorf("invalid level or k vector: %+v", c)
	}
	for i, x := range c.Shifts {
		if x <= 0 || x >= n/2 || slices.Contains(c.Shifts[:i], x) {
			return fmt.Errorf("invalid or duplicate rotation %d", x)
		}
	}
	for _, k := range c.K {
		if k < 0 || k > c.Level || k > 4 {
			return fmt.Errorf("unmeasurable k=%d", k)
		}
	}
	return nil
}

// Inputs, evaluation keys and the evaluator are built outside the clock. The
// returned closure runs the allocating production ABI, including Giant's copy.
type CalibrationFixture struct {
	p      ckks.Parameters
	e      *ckks.Evaluator
	ct     *rlwe.Ciphertext
	op     string
	shifts []int
}

// NewCalibrationPair uses the same keys, but separately encrypted timing and
// holdout inputs. The caller owns the pair and uses it serially.
func NewCalibrationPair(p ckks.Parameters, c CalibrationCall) ([2]*CalibrationFixture, error) {
	var pair [2]*CalibrationFixture
	if err := c.Validate(p.N(), p.QCount()); err != nil {
		return pair, err
	}
	gen := rlwe.NewKeyGenerator(p)
	sk := gen.GenSecretKeyNew()
	var rlk *rlwe.RelinearizationKey
	var gks []*rlwe.GaloisKey
	if c.Op == "relin" {
		rlk = gen.GenRelinearizationKeyNew(sk)
	}
	for _, x := range c.Shifts {
		gks = append(gks, gen.GenGaloisKeyNew(p.GaloisElement(x), sk))
	}
	e := ckks.NewEvaluator(p, rlwe.NewMemEvaluationKeySet(rlk, gks...))
	enc := ckks.NewEncoder(p)
	encrypt := rlwe.NewEncryptor(p, sk)
	for j := range pair {
		pt := ckks.NewPlaintext(p, c.Level)
		slots := make([]float64, p.MaxSlots())
		for i := range slots {
			slots[i] = float64((17*i+13*j)%59-29) / 64
		}
		if err := enc.Encode(slots, pt); err != nil {
			return pair, err
		}
		ct, err := encrypt.EncryptNew(pt)
		if err != nil {
			return pair, err
		}
		if c.Op == "relin" {
			ct, err = e.MulNew(ct, ct)
			if err != nil {
				return pair, err
			}
		}
		pair[j] = &CalibrationFixture{p, e, ct, c.Op, slices.Clone(c.Shifts)}
	}
	return pair, nil
}

// Each call returns ordered outputs. Materializing a diagnostic slice or
// hashing outputs is deliberately left outside the timed production call.
type CalibrationOutput struct {
	Single *rlwe.Ciphertext
	Group  map[int]*rlwe.Ciphertext
}

func (o CalibrationOutput) At(shift int) *rlwe.Ciphertext {
	if o.Single != nil {
		return o.Single
	}
	return o.Group[shift]
}

func (f *CalibrationFixture) Bind(c CalibrationCall) (func() (CalibrationOutput, error), error) {
	if err := c.Validate(f.p.N(), f.p.QCount()); err != nil {
		return nil, err
	}
	if c.Op != f.op || c.Level != f.ct.Level() || !slices.Equal(c.Shifts, f.shifts) {
		return nil, fmt.Errorf("wrong calibration fixture")
	}
	site := Site{ID: "calibration", Level: c.Level, K: c.K[0]}
	switch c.Op {
	case "relin":
		return func() (CalibrationOutput, error) {
			x, e := RelinearizeNew(f.e, f.ct, site)
			return CalibrationOutput{Single: x}, e
		}, nil
	case "giant":
		return func() (CalibrationOutput, error) {
			x, e := Giant(f.e, f.ct.CopyNew(), c.Shifts[0], site)
			return CalibrationOutput{Single: x}, e
		}, nil
	default:
		sites := make(map[int]Site, len(c.Shifts))
		for i, x := range c.Shifts {
			sites[x] = Site{ID: fmt.Sprint(x), Level: c.Level, K: c.K[i]}
		}
		return func() (CalibrationOutput, error) {
			x, e := Babies(f.e, f.ct, c.Shifts, sites)
			return CalibrationOutput{Group: x}, e
		}, nil
	}
}

// Independent, dense big.Int CRT oracle; never called from a timed closure.
func denseCalibrationInput(rq *ring.Ring, ct *rlwe.Ciphertext, k int) *rlwe.Ciphertext {
	cp := ct.CopyNew()
	if k == 0 {
		return cp
	}
	x := cp.Value[cp.Degree()]
	rq.INTT(x, x)
	z := make([]*big.Int, rq.N())
	rq.PolyToBigint(x, 1, z)
	d := rq.ModulusAtLevel[k-1]
	half := new(big.Int).Rsh(new(big.Int).Set(d), 1)
	var r big.Int
	for _, a := range z {
		r.Mod(a, d)
		if r.Cmp(half) > 0 {
			r.Sub(&r, d)
		}
		a.Sub(a, &r)
	}
	rq.SetCoefficientsBigint(z, x)
	rq.NTT(x, x)
	return cp
}

// VerifyCalibration consumes every output and checks ciphertext AND metadata
// against native Lattigo on the independently projected input. No inference
// about slot accuracy or security is made from the calibration fixture.
func (f *CalibrationFixture) VerifyCalibration(c CalibrationCall, out CalibrationOutput) error {
	for i, k := range c.K {
		ref := denseCalibrationInput(f.p.RingQ().AtLevel(c.Level), f.ct, k)
		var want *rlwe.Ciphertext
		var err error
		var shift int
		switch c.Op {
		case "relin":
			want, err = f.e.RelinearizeNew(ref)
		case "giant":
			shift = c.Shifts[i]
			want, err = f.e.RotateNew(ref, shift)
		case "baby":
			shift = c.Shifts[i]
			var m map[int]*rlwe.Ciphertext
			m, err = f.e.RotateHoistedNew(ref, []int{shift})
			want = m[shift]
		}
		if err != nil {
			return err
		}
		got := out.At(shift)
		if got == nil || !want.Equal(got) {
			return fmt.Errorf("dense reference mismatch op=%s shift=%d k=%d", c.Op, shift, k)
		}
	}
	return nil
}
