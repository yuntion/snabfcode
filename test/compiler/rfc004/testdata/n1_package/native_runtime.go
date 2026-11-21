package planned

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func relin(e *ckks.Evaluator, x *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	return e.RelinearizeNew(x)
}
