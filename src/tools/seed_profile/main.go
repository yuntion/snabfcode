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
	alpha := flag.Int("p-count", 1, "provisional number of special limbs; not a security decision")
	flag.Parse()
	if *n < 10 || *n > 17 || *count < 2 || *count > 32 || *alpha < 1 || *alpha > 4 {
		fmt.Fprintln(os.Stderr, "invalid seed ring/chain")
		os.Exit(2)
	}
	q := make([]int, *count)
	for i := range q {
		q[i] = 40
	}
	q[0] = 50
	aux := make([]int, *alpha)
	for i := range aux {
		aux[i] = 55
	}
	p, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: *n, LogQ: q, LogP: aux, LogDefaultScale: 40})
	if err != nil {
		panic(err)
	}
	if err = json.NewEncoder(os.Stdout).Encode(map[string]any{"logN": *n, "Q": p.Q(), "P": p.P(), "logDefaultScale": 40, "encryptionType": "pk", "encryptionTechnique": "standard"}); err != nil {
		panic(err)
	}
}
