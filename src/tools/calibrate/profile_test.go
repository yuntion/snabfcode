package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/gadget"
)

func TestRawUnitsAndQuantiles(t *testing.T) {
	a, m, i := summarize([]int64{10, 1, 5, 3})
	if !slices.Equal(a, []string{"0.01000", "0.00100", "0.00500", "0.00300"}) || m != "0.00400" || i != "0.00375" {
		t.Fatalf("raw=%v median=%s IQR=%s", a, m, i)
	}
	a1, e := digest(map[string]any{"z": "<&", "a": []int{0, 1}})
	if e != nil {
		t.Fatal(e)
	}
	a2, e := digest(map[string]any{"a": []int{0, 1}, "z": "<&"})
	if e != nil || a1 != a2 {
		t.Fatal("unstable fingerprint")
	}
	a3, _ := digest(map[string]any{"z": "<&", "a": []int{1, 0}})
	if a3 == a1 {
		t.Fatal("array order omitted")
	}
	if _, e = digest(1.5); e == nil {
		t.Fatal("floating metadata unexpectedly allowed")
	}
}
func TestManifestAndWholeSite(t *testing.T) {
	p, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 8, LogQ: []int{25, 36, 36}, LogP: []int{39, 38}, LogDefaultScale: 35})
	if e != nil {
		t.Fatal(e)
	}
	c := benchCase{shape{p.N(), p.Q(), p.P()}, 2, "baby", []int{3, 1}, [][]int{{0, 0}, {1, 2}, {1, 1}}}
	m := map[string]any{"cases": []benchCase{c}}
	data, e := json.Marshal(m)
	if e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(t.TempDir(), "manifest.json")
	if e = os.WriteFile(path, data, 0600); e != nil {
		t.Fatal(e)
	}
	cs, e := cases(path)
	if e != nil || len(cs) != 1 || !slices.Equal(cs[0].Q, c.Q) {
		t.Fatal(cs, e)
	}
	rows, e := measure(cs[0], 0, 1, 3, 2)
	if e != nil {
		t.Fatal(e)
	}
	if len(rows) != 3 {
		t.Fatal("missing group patterns")
	}
	for i, r := range rows {
		if len(r.Samples) != 3 || len(r.Holdout) != 2 || !slices.Equal(r.Rows, []int{2, 1}) || !slices.Equal(r.K, c.Patterns[i]) {
			t.Fatalf("bad timing record: %+v", r)
		}
	}
	obs := rows[1].Observations
	if len(rows[0].Observations) != 0 || len(obs) != 2 || obs[0].K != 1 || obs[0].Skipped != 0 || obs[1].K != 2 || obs[1].Skipped != 1 || gadget.Observe != nil {
		t.Fatalf("independent diagnostics: %+v", obs)
	}
	if _, e = digest(rows); e != nil {
		t.Fatal(e)
	}
	for _, bad := range [][][]int{{{1, 1}}, {{0, 0}, {0, 0}}, {{0, 0}, {3, 0}}} {
		c.Patterns = bad
		if _, e = measure(c, 0, 1, 1, 1); e == nil {
			t.Fatal("accepted bad patterns", bad)
		}
	}
	if e = os.WriteFile(path, []byte(`{"cases":[{"unknown":1}]}`), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e = cases(path); e == nil {
		t.Fatal("unknown manifest field")
	}
}
