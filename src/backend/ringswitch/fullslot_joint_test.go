package ringswitch

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/tower"
)

func linearEvidence(t *testing.T, file string, data []byte) {
	t.Helper()
	if dir := os.Getenv("SNABF_A12_EVIDENCE"); dir != "" {
		if err := os.WriteFile(filepath.Join(dir, file), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
}

func jointLinearRoutes(t *testing.T, exe string, dst ckks.Parameters, blocks [2][2]tower.Diagonals[float64], downID string) (tower.FullSlotRoutes, []uint64, int) {
	t.Helper()
	profile := map[string]any{"logN": dst.LogN(), "Q": dst.Q(), "P": dst.P()}
	identity, err := json.Marshal([]any{"split-target-v1", profile, downID})
	if err != nil {
		t.Fatal(err)
	}
	scope := fmt.Sprintf("%x", sha256.Sum256(identity))
	l := dst.MaxLevel()
	bytesPerKey := rlwe.NewGaloisKey(dst, rlwe.EvaluationKeyParameters{LevelQ: &l}).BinarySize()
	var regions []map[string]any
	for o, row := range blocks {
		for i, b := range row {
			if len(b) == 0 {
				t.Fatal("fixture must price all four nonzero children")
			}
			regions = append(regions, map[string]any{"apply_id": fmt.Sprintf("target.%d.%d", o, i), "level": l, "log_slots": dst.LogMaxSlots(), "mask_scale": dst.Q()[l], "diagonals": slices.Sorted(maps.Keys(b))})
		}
	}
	input, err := json.Marshal(map[string]any{"scope": scope, "profile": profile, "regions": regions, "bytes_per_key": bytesPerKey, "costs": map[string]float64{"hoisted_per_active_limb": .625, "giant_per_active_limb": 1, "key_units_per_mib": 1}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe)
	cmd.Stdin = bytes.NewReader(input)
	var diag bytes.Buffer
	cmd.Stderr = &diag
	answer, err := cmd.Output()
	if err != nil {
		t.Fatalf("native Pass 1: %v (%s)", err, diag.String())
	}
	linearEvidence(t, "pass1-request.json", input)
	linearEvidence(t, "pass1-answer.json", answer)
	routes, keys, err := tower.BindFullSlotJoint(answer, dst, l, blocks, scope, bytesPerKey)
	if err != nil {
		t.Fatal(err)
	}
	type reply struct {
		Candidates [][]struct {
			Terms  [][]int
			Galois []uint64
		}
		Joint struct {
			Selected []int
			KeyBytes int `json:"key_bytes"`
			Keys     []struct {
				Scope  string
				Galois uint64
				Refs   int
			}
		}
	}
	var orig reply
	if err := json.Unmarshal(answer, &orig); err != nil {
		t.Fatal(err)
	}
	if len(orig.Joint.Selected) != 4 || len(keys) != 2 || len(orig.Joint.Keys) != len(keys) {
		t.Fatalf("unexpected joint selection: %+v", orig.Joint)
	}
	for _, tc := range []struct {
		name    string
		corrupt func(*reply)
	}{
		{"missing-child", func(v *reply) { v.Joint.Selected = v.Joint.Selected[:3] }},
		{"wrong-family", func(v *reply) { v.Joint.Keys[0].Scope = "other" }},
		{"wrong-refcount", func(v *reply) { v.Joint.Keys[0].Refs++ }},
		{"wrong-bytes", func(v *reply) { v.Joint.KeyBytes++ }},
		{"bad-diagonal", func(v *reply) { v.Candidates[0][v.Joint.Selected[0]].Terms[0][1] = dst.MaxSlots() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var bad reply
			if err := json.Unmarshal(answer, &bad); err != nil {
				t.Fatal(err)
			}
			tc.corrupt(&bad)
			raw, err := json.Marshal(bad)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := tower.BindFullSlotJoint(raw, dst, l, blocks, scope, bytesPerKey); err == nil {
				t.Fatal("tampered joint answer accepted")
			} else {
				t.Log(err)
			}
		})
	}
	return routes, keys, orig.Joint.KeyBytes
}
