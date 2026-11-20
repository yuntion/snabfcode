package epochs

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Generated ports describe the intended call, not observations of a run.
type ExactPort struct {
	ID, Output, Call, Metadata, Effective, PayloadSHA256 string
	N, Level, Precision, Count                           int
}

func exactScale(text string) (rlwe.Scale, error) {
	if len(text) == 0 || len(text) > 10000 {
		return rlwe.Scale{}, fmt.Errorf("missing or oversized scale")
	}
	for _, c := range text {
		if c != '/' && (c < '0' || c > '9') {
			return rlwe.Scale{}, fmt.Errorf("expected decimal numerator/denominator")
		}
	}
	q, ok := new(big.Rat).SetString(text)
	if !ok || q.Sign() <= 0 || q.RatString() != text {
		return rlwe.Scale{}, fmt.Errorf("invalid exact scale")
	}
	x := new(big.Float).SetPrec(rlwe.ScalePrecision).SetRat(q)
	back, _ := x.Rat(nil)
	if q.Cmp(back) != 0 {
		return rlwe.Scale{}, fmt.Errorf("scale not exact at 128 bits")
	}
	return rlwe.NewScale(x), nil
}

func payloadDigest(values []float64) string {
	h := sha256.New()
	var word [8]byte
	for _, x := range values {
		binary.BigEndian.PutUint64(word[:], math.Float64bits(x))
		h.Write(word[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func validateExactPorts(s Program) (map[string]rlwe.Scale, map[int]rlwe.Scale, error) {
	bad := func() (map[string]rlwe.Scale, map[int]rlwe.Scale, error) {
		return nil, nil, fmt.Errorf("missing or stale encoding scale binding")
	}
	if len(s.Ports) == 0 {
		return bad()
	}
	out := make(map[string]rlwe.Scale, len(s.Ports))
	scales := make(map[int]rlwe.Scale)
	plain := make(map[int]PlainBinding, len(s.Plain))
	for _, p := range s.Plain {
		if _, exists := plain[p.Index]; exists {
			return bad()
		}
		plain[p.Index] = p
	}
	for i, p := range s.Ports {
		if _, exists := out[p.ID]; p.ID == "" || exists {
			return bad()
		}
		v, err := exactScale(p.Output)
		if err != nil {
			return bad()
		}
		out[p.ID] = v
		if (p.Call == "source") != (i == 0) {
			return bad()
		}
		if p.Call == "" {
			continue
		}
		meta, err := exactScale(p.Metadata)
		if err != nil {
			return bad()
		}
		scales[i] = meta
		if p.Call == "source" {
			if p.N != s.Source || p.Metadata != p.Output || p.Count != 0 || p.PayloadSHA256 != payloadDigest(nil) {
				return bad()
			}
			continue
		}
		arg, ok := plain[i]
		if !ok || (p.Call != "coefficient" && p.Call != "mask" && p.Call != "add_plain") || arg.Level != p.Level || arg.N != p.N ||
			len(arg.Values) == 0 || len(arg.Values) != p.Count || p.PayloadSHA256 != payloadDigest(arg.Values) {
			return bad()
		}
		delete(plain, i)
	}
	if len(plain) != 0 {
		return bad()
	}
	return out, scales, nil
}

// Check at the actual batched Encode call. All current calls pass []float64,
// even a one-element constant tiled over the slots. Do not select the branch
// by coefficient count or by metadata's significant bits.
func checkEncoderPort(p ExactPort, pt *rlwe.Plaintext, enc *ckks.Encoder) error {
	if !pt.IsBatched || pt.Level() != p.Level || pt.N() != p.N || pt.Scale.Mod != nil ||
		pt.Scale.Value.Prec() != rlwe.ScalePrecision || int(enc.Prec()) != p.Precision {
		return fmt.Errorf("wrong encoder call: %s", p.ID)
	}
	meta, _ := pt.Scale.Value.Rat(nil)
	effective := meta
	if enc.Prec() <= 53 {
		effective = new(big.Rat).SetFloat64(pt.Scale.Float64())
	}
	if effective == nil || meta.RatString() != p.Metadata || effective.RatString() != p.Effective {
		return fmt.Errorf("wrong encoder scale: %s", p.ID)
	}
	return nil
}
