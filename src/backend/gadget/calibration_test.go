package gadget

import (
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"testing"
)

func TestCalibrationABIs(t *testing.T) {
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 8, LogQ: []int{25, 36, 36, 37}, LogP: []int{39, 38}, LogDefaultScale: 35})
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range []CalibrationCall{{"relin", 2, nil, []int{2}}, {"giant", 2, []int{7}, []int{1}}, {"baby", 2, []int{3, 1, 2}, []int{0, 1, 2}}} {
		t.Run(a.Op, func(t *testing.T) {
			pair, e := NewCalibrationPair(p, a)
			if e != nil {
				t.Fatal(e)
			}
			f, e := pair[0].Bind(a)
			if e != nil {
				t.Fatal(e)
			}
			out, e := f()
			if e != nil {
				t.Fatal(e)
			}
			if e = pair[0].VerifyCalibration(a, out); e != nil {
				t.Fatal(e)
			}
			if e = pair[1].VerifyCalibration(a, out); e == nil {
				t.Fatal("accepted output from the training cohort for the independent holdout")
			}
			bad := a
			bad.Level--
			if _, e = pair[0].Bind(bad); e == nil {
				t.Fatal("stale level accepted")
			}
		})
	}
	for _, a := range []CalibrationCall{{"baby", 2, []int{1, 1}, []int{0, 0}}, {"giant", 2, []int{0}, []int{0}}, {"relin", 2, []int{1}, []int{0}}, {"relin", 2, nil, []int{3}}, {"down", 2, nil, []int{0}}} {
		if a.Validate(p.N(), p.QCount()) == nil {
			t.Fatalf("accepted %+v", a)
		}
	}
}
