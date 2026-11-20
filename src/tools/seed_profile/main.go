// A provisional HEIR typing profile, never evidence of parameter admission.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {
	n := flag.Int("log-n", 14, "provisional ring log dimension")
	count := flag.Int("q-count", 8, "provisional number of Q limbs")
	flag.Parse()
	if *n < 10 || *n > 17 || *count < 2 || *count > 32 {
		fmt.Fprintln(os.Stderr, "invalid seed ring/chain")
		os.Exit(2)
	}
	q := make([]int, *count)
	for i := range q {
		q[i] = 40
	}
	q[0] = 50
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: *n, LogQ: q, LogP: []int{55}, LogDefaultScale: 40})
	if err != nil {
		panic(err)
	}
	if err = json.NewEncoder(os.Stdout).Encode(map[string]any{"logN": *n, "Q": p.Q(), "P": p.P(), "logDefaultScale": 40, "encryptionType": "pk", "encryptionTechnique": "standard"}); err != nil {
		panic(err)
	}
}
