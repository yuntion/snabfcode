// External observation driver: never opens the compile-time plan for writing.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"snabf.local/compiler/planned"
	"time"
)

type input struct {
	Seal    string      `json:"emit_sha256"`
	Source  string      `json:"source_sha256"`
	Vectors [][]float64 `json:"inputs"`
}

func run() error {
	var in input
	if err := json.NewDecoder(os.Stdin).Decode(&in); err != nil {
		return err
	}
	if in.Seal != planned.PlanSHA256 || len(in.Vectors) != 64 {
		return fmt.Errorf("wrong checked plan/input count")
	}
	r, err := planned.New()
	if err != nil {
		return err
	}
	swkBytes := 0
	for _, k := range r.Keys {
		if k.Role == "down" {
			swkBytes += k.Bytes
		}
	}
	out := json.NewEncoder(os.Stdout)
	for i, x := range in.Vectors {
		encrypted, err := r.Encrypt(x)
		if err != nil {
			return err
		}
		start := time.Now()
		y, err := r.Eval(encrypted)
		elapsed := time.Since(start)
		if err != nil {
			return err
		}
		decoded, err := r.Decode(y)
		if err != nil {
			return err
		}
		scale, _ := y.Scale.Value.Rat(nil)
		row := map[string]any{"trial": i, "source_sha256": in.Source, "emit_sha256": planned.PlanSHA256,
			"x": x, "decoded": decoded, "ring": y.Value[0].N(), "level": y.Level(),
			"scale_f64": y.Scale.Float64(), "scale_value": scale.RatString(),
			"swk_bytes": swkBytes, "eval_ns": elapsed.Nanoseconds()}
		if err := out.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
