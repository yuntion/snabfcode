// Measure serialized key objects at the exact Q/P used by each proposal.
package main

import (
	"encoding/json"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math/bits"
	"os"
)

func main() {
	var data struct {
		Ring, Scale int
		Q, P        []uint64
	}
	f, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	defer f.Close()
	if err = json.NewDecoder(f).Decode(&data); err != nil {
		panic(err)
	}
	params := func(n, l int) ckks.Parameters {
		p, e := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: bits.TrailingZeros(uint(n)), Q: data.Q[:l+1], P: data.P, LogDefaultScale: data.Scale})
		if e != nil {
			panic(e)
		}
		return p
	}
	large := params(data.Ring, len(data.Q)-1)
	sizes := map[string]any{}
	for n := data.Ring / 2; n >= 1024; n /= 2 {
		for l := 1; l < len(data.Q); l++ {
			small := params(n, l)
			lp := len(data.P) - 1
			g := rlwe.NewGaloisKey(small)
			sw := rlwe.NewEvaluationKey(large, rlwe.EvaluationKeyParameters{LevelQ: &l, LevelP: &lp})
			sizes[fmt.Sprintf("%d:%d", n, l)] = map[string]int{"key_bytes": g.BinarySize(), "switch_bytes": sw.BinarySize(), "key_level_q": sw.LevelQ(), "key_level_p": sw.LevelP()}
		}
	}
	if err = json.NewEncoder(os.Stdout).Encode(sizes); err != nil {
		panic(err)
	}
}
