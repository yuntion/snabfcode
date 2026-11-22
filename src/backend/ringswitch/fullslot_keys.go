package ringswitch

import (
	"crypto/sha256"
	"encoding"
	"encoding/binary"
	"fmt"
	"maps"
	"slices"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type FullSlotKeys struct {
	Fold   *rlwe.GaloisKey
	Down   *rlwe.EvaluationKey
	Target []*rlwe.GaloisKey
}

// Serialized bytes of distinct keys, not multiplied by their number of uses.
type FullSlotKeyInventory struct {
	FoldBytes, DownBytes, TargetBytes, TargetKeys int
}

// Commitment must be pinned at trusted provisioning, where the source and
// destination secrets are known. Recomputing the expected pin from untrusted
// delivery bytes would not establish the key family. Ciphertexts don't carry
// an authenticated secret ID; the caller still owns the source input binding.
func (k FullSlotKeys) Commitment(src, dst ckks.Parameters, sourceLevel int) ([32]byte, error) {
	id, _, err := k.inspect(src, dst, sourceLevel)
	return id, err
}

func (k FullSlotKeys) inspect(src, dst ckks.Parameters, sourceLevel int) ([32]byte, FullSlotKeyInventory, error) {
	var id [32]byte
	var n FullSlotKeyInventory
	l := sourceLevel - 1
	if l < 1 || sourceLevel > src.MaxLevel() || l > dst.MaxLevel() || src.N() != 2*dst.N() || k.Fold == nil || k.Fold.GaloisElement != src.GaloisElement(dst.MaxSlots()) || k.Fold.NthRoot != src.RingQ().NthRoot() || !splitKey(src, &k.Fold.EvaluationKey, l) || !splitKey(src, k.Down, l) {
		return id, n, fmt.Errorf("invalid source fold or switching key")
	}
	rot := map[uint64]*rlwe.GaloisKey{}
	for _, key := range k.Target {
		if key == nil || key.GaloisElement <= 1 || key.GaloisElement >= dst.RingQ().NthRoot() || key.GaloisElement%2 != 1 || key.NthRoot != dst.RingQ().NthRoot() || rot[key.GaloisElement] != nil || !splitKey(dst, &key.EvaluationKey, l) || key.LevelQ() != l {
			return id, n, fmt.Errorf("invalid or duplicate target rotation key")
		}
		rot[key.GaloisElement] = key
	}
	h := sha256.New()
	h.Write([]byte("snabf.full-slot-linear.keys.v1\x00"))
	h.Write(binary.LittleEndian.AppendUint64(nil, uint64(sourceLevel)))
	// Fixed-size digests in role order; target key encodings include the
	// Galois element. Parameter encodings include the actual ordered Q/P.
	bind := func(v encoding.BinaryMarshaler) (int, error) {
		raw, err := v.MarshalBinary()
		if err != nil {
			return 0, err
		}
		digest := sha256.Sum256(raw)
		h.Write(digest[:])
		return len(raw), nil
	}
	if _, err := bind(src); err != nil {
		return id, n, err
	}
	if _, err := bind(dst); err != nil {
		return id, n, err
	}
	var err error
	if n.FoldBytes, err = bind(k.Fold); err != nil {
		return id, n, err
	}
	if n.DownBytes, err = bind(k.Down); err != nil {
		return id, n, err
	}
	for _, g := range slices.Sorted(maps.Keys(rot)) {
		b, err := bind(rot[g])
		if err != nil {
			return id, n, err
		}
		n.TargetBytes += b
		n.TargetKeys++
	}
	copy(id[:], h.Sum(nil))
	return id, n, nil
}
