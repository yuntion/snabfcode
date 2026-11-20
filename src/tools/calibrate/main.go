// Command snabf-calibrate measures complete production gadget invocations.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/bits"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"time"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/gadget"
)

func cases(path string) ([]benchCase, error) {
	if path == "" {
		p, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 13, LogQ: []int{25, 36, 36}, LogP: []int{39, 38}, LogDefaultScale: 36})
		if e != nil {
			return nil, e
		}
		s := shape{p.N(), p.Q(), p.P()}
		return []benchCase{{s, 2, "relin", []int{}, [][]int{{0}, {1}, {2}}}, {s, 2, "giant", []int{1}, [][]int{{0}, {1}, {2}}}, {s, 2, "baby", []int{1, 2}, [][]int{{0, 0}, {1, 1}, {2, 2}, {0, 1}}}}, nil
	}
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	var m struct {
		Cases []benchCase `json:"cases"`
	}
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if e = d.Decode(&m); e != nil {
		return nil, e
	}
	if d.Decode(new(any)) != io.EOF || len(m.Cases) == 0 {
		return nil, fmt.Errorf("empty manifest or trailing data")
	}
	return m.Cases, nil
}

type invocation func() (gadget.CalibrationOutput, error)

func clock(f invocation) (gadget.CalibrationOutput, int64, error) {
	runtime.GC()
	prev := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(prev)
	t := time.Now()
	x, e := f()
	ns := time.Since(t).Nanoseconds()
	return x, ns, e
}

func measure(c benchCase, id, warm, samples, hold int) ([]point, error) {
	if c.N < 16 || c.N > 65536 || c.N&(c.N-1) != 0 || len(c.Q) == 0 || len(c.Q) > 64 || len(c.P) < 1 || len(c.P) > 4 || len(c.Patterns) == 0 {
		return nil, fmt.Errorf("unsupported case shape")
	}
	var calls []gadget.CalibrationCall
	native := false
	seen := map[string]bool{}
	for _, ks := range c.Patterns {
		a := gadget.CalibrationCall{Op: c.Op, Level: c.Ell, Shifts: c.Shifts, K: ks}
		if e := a.Validate(c.N, len(c.Q)); e != nil {
			return nil, e
		}
		key := fmt.Sprint(ks)
		if seen[key] {
			return nil, fmt.Errorf("duplicate pattern %v", ks)
		}
		seen[key] = true
		native = native || slices.Max(ks) == 0
		calls = append(calls, a)
	}
	if !native {
		return nil, fmt.Errorf("cost_profile_missing_site: all-native pattern required")
	}
	// Pass 3 admits distinct, ordered, sub-60-bit NTT primes. Lattigo
	// checks primality too; reject incompatible shapes before keygen.
	used := map[uint64]bool{}
	for _, basis := range [][]uint64{c.Q, c.P} {
		for _, z := range basis {
			if z < 3 || z >= 1<<60 || z%uint64(2*c.N) != 1 || used[z] {
				return nil, fmt.Errorf("invalid or repeated concrete NTT modulus %d", z)
			}
			used[z] = true
		}
	}
	// The scale is metadata for these integer kernels; choose a representable
	// fixture scale, not a claim that a user's graph passes a precision test.
	logScale := min(36, bits.Len64(c.Q[0])-3)
	if logScale < 1 {
		return nil, fmt.Errorf("invalid q0")
	}
	p, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: bits.TrailingZeros(uint(c.N)), Q: c.Q, P: c.P, LogDefaultScale: logScale})
	if e != nil {
		return nil, e
	}
	if gadget.Observe != nil {
		return nil, fmt.Errorf("timed evaluator requires a disabled observer")
	}
	pair, e := gadget.NewCalibrationPair(p, calls[0])
	if e != nil {
		return nil, e
	}
	out := make([]point, len(calls))
	raw := make([][]int64, len(calls))
	holdout := make([][]int64, len(calls))
	for stage, fixture := range pair {
		fns := make([]invocation, len(calls))
		for i, a := range calls {
			fns[i], e = fixture.Bind(a)
			if e != nil {
				return nil, e
			}
		}
		count := samples
		if stage == 1 {
			count = hold
		}
		for round := 0; round < warm+count; round++ {
			for offset := range calls {
				i := (round + offset) % len(calls)
				x, ns, e := clock(fns[i])
				if e != nil {
					return nil, e
				}
				for j := range calls[i].K {
					shift := 0
					if len(c.Shifts) > 0 {
						shift = c.Shifts[j]
					}
					v := x.At(shift)
					if ns <= 0 || v == nil || v.Level() != c.Ell || v.Degree() != 1 {
						return nil, fmt.Errorf("invalid timed result")
					}
				}
				// Validate the actual unobserved ABI on an independent input,
				// after the final holdout clock; no oracle is timed.
				if stage == 1 && round == warm+count-1 {
					if e = fixture.VerifyCalibration(calls[i], x); e != nil {
						return nil, e
					}
				}
				runtime.KeepAlive(x)
				if round >= warm {
					if stage == 0 {
						raw[i] = append(raw[i], ns)
					} else {
						holdout[i] = append(holdout[i], ns)
					}
				}
			}
		}
	}
	for i, a := range calls {
		row := point{Shape: id, Ell: c.Ell, Op: c.Op, Shifts: append([]int{}, c.Shifts...), K: a.K, Observations: []observation{}}
		for j := 0; j <= c.Ell; j += len(c.P) {
			row.Rows = append(row.Rows, min(len(c.P), c.Ell+1-j))
		}
		row.Samples, row.Median, row.IQR = summarize(raw[i])
		row.Holdout, _, _ = summarize(holdout[i])
		f, _ := pair[1].Bind(a)
		// Separate, untimed diagnostic: native k=0 has no observer event.
		x, e := func() (gadget.CalibrationOutput, error) {
			defer func() { gadget.Observe = nil }()
			gadget.Observe = func(v gadget.Event) {
				row.Observations = append(row.Observations, observation{v.TruncatedLimbsK, v.FullBlocksSkipped, v.MeasuredSingleLimbNTTsSaved})
			}
			return f()
		}()
		if e != nil {
			return nil, e
		}
		if e = pair[1].VerifyCalibration(a, x); e != nil {
			return nil, e
		}
		out[i] = row
	}
	return out, nil
}

func run() error {
	manifest := flag.String("manifest", "", "JSON cases with concrete q/p, ell, op, shifts and k patterns")
	out := flag.String("out", "hw_profile.json", "new output (never overwrites)")
	rev := flag.String("bench-rev", "", "revision for a build without VCS metadata")
	warm := flag.Int("warmup", 5, "warmups per pattern per cohort")
	samples := flag.Int("samples", 30, "training samples per pattern")
	hold := flag.Int("holdout", 5, "independent-input validation samples per pattern")
	idOnly := flag.Bool("print-hardware-id", false, "print the reserved CPU fingerprint and exit")
	flag.Parse()
	if flag.NArg() != 0 || *warm < 1 || *warm > 10000 || *samples < 1 || *samples > 10000 || *hold < 1 || *hold > 10000 {
		return fmt.Errorf("invalid arguments or sampling counts")
	}
	runtime.GOMAXPROCS(1)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	hw, id, e := hardware()
	if e != nil {
		return e
	}
	if *idOnly {
		fmt.Println(id)
		return nil
	}
	tc, r, e := toolchain(*rev)
	if e != nil {
		return e
	}
	list, e := cases(*manifest)
	if e != nil {
		return e
	}
	p := profile{Schema: "snabf-gadget-sites-v1", Units: "us", ABI: "direct-go-expanded-pure-rns-ntt-v1", HardwareID: id, Hardware: hw, Toolchain: tc, BenchRev: r, Created: time.Now().UTC().Format(time.RFC3339Nano), Samples: *samples, Warmup: *warm, Holdout: *hold, Quantiles: "type7-from-raw-us"}
	shapes := map[string]int{}
	seen := map[string]bool{}
	for j, c := range list {
		s, e := digest(c.shape)
		if e != nil {
			return fmt.Errorf("case %d basis: %w", j, e)
		}
		idx, ok := shapes[s]
		if !ok {
			idx = len(p.Shapes)
			p.Shapes = append(p.Shapes, c.shape)
			shapes[s] = idx
		}
		key := fmt.Sprintf("%d:%d:%s:%v", idx, c.Ell, c.Op, c.Shifts)
		if seen[key] {
			return fmt.Errorf("duplicate call family: %s", key)
		}
		seen[key] = true
		rows, e := measure(c, idx, *warm, *samples, *hold)
		if e != nil {
			return fmt.Errorf("case %d (%s): %w", j, key, e)
		}
		p.Sites = append(p.Sites, rows...)
	}
	p.Digest, e = digest(p)
	if e != nil {
		return e
	}
	b, e := json.MarshalIndent(p, "", "  ")
	if e != nil {
		return e
	}
	f, e := os.OpenFile(*out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if e != nil {
		return e
	}
	if _, e = f.Write(append(b, '\n')); e != nil {
		f.Close()
		return e
	}
	return f.Close()
}
func main() {
	if e := run(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
