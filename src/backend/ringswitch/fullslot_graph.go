package ringswitch

import (
	"crypto/sha256"
	"encoding"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"snabf.local/compiler/src/backend/tower"
)

// GraphKeys contains evaluation material only. Pin its commitment at trusted
// provisioning, where both secrets and the ciphertext input domain are known.
type GraphKeys struct {
	Relin  *rlwe.RelinearizationKey
	Source []*rlwe.GaloisKey
	Split  *FullSlotKeys
}

func (k GraphKeys) Commitment(s, d ckks.Parameters) (id [32]byte, err error) {
	if k.Relin == nil {
		return id, fmt.Errorf("missing source relin")
	}
	h := sha256.New()
	h.Write([]byte("snabf.a2b.graph-keys.v1\x00"))
	bind := func(x encoding.BinaryMarshaler) error {
		raw, e := x.MarshalBinary()
		if e == nil {
			a := sha256.Sum256(raw)
			h.Write(a[:])
		}
		return e
	}
	for _, v := range []encoding.BinaryMarshaler{s, d, k.Relin} {
		if err = bind(v); err != nil {
			return id, err
		}
	}
	rot := map[uint64]*rlwe.GaloisKey{}
	for _, v := range k.Source {
		if v == nil || rot[v.GaloisElement] != nil {
			return id, fmt.Errorf("missing or repeated source rotation")
		}
		rot[v.GaloisElement] = v
	}
	for _, g := range slices.Sorted(maps.Keys(rot)) {
		if err = bind(rot[g]); err != nil {
			return id, err
		}
	}
	if k.Split != nil {
		p, e := k.Split.Commitment(s, d, 2)
		if e != nil {
			return id, e
		}
		h.Write([]byte{1})
		h.Write(p[:])
	} else {
		h.Write([]byte{0})
	}
	copy(id[:], h.Sum(nil))
	return id, nil
}

type graphLedger struct {
	Cut  string `json:"cut_scale"`
	Sink string `json:"sink_scale"`
}
type childPort struct {
	Start, Used, N, Level int
	Q, P                  []uint64
	ActiveQ               []uint64 `json:"active_q"`
	Scale                 string
	Domain                string `json:"key_domain"`
}
type checkedGraph struct {
	Schema, Selected string
	Source           string `json:"source_sha256"`
	Basis            struct {
		Q, P []uint64
		D    int
	}
	Input      int `json:"input_width"`
	Output     int `json:"output_width"`
	Keep, Down graphLedger
	Fold       struct {
		Offsets [2]int
		Galois  uint64
		SlotMap string `json:"slot_map_sha256"`
	}
	Children []struct {
		Out, In int
		Route   *tower.Route
	}
	KeepRoute  tower.Route `json:"keep_route"`
	SplitPorts []childPort `json:"split_outputs"`
	DownPorts  []childPort `json:"outputs_down"`
}
type PreparedGraph struct {
	s, d   ckks.Parameters
	p      checkedGraph
	ev     *ckks.Evaluator
	linear *FullSlotLinear
	keep   tower.LinearTransformation
}

// PrepareCheckedGraph consumes the bytes and independent digest constants
// emitted by snabf-lower-ckks. It is not an alternative admission checker.
func PrepareCheckedGraph(source, program []byte, sourceID, planID string, keys GraphKeys, pin [32]byte) (*PreparedGraph, error) {
	hash := func(v []byte) string { return fmt.Sprintf("%x", sha256.Sum256(v)) }
	if hash(program) != planID || hash(source) != sourceID {
		return nil, fmt.Errorf("substituted emitted program/source")
	}
	var p checkedGraph
	var original struct {
		Input     int `json:"input_width"`
		Output    int `json:"output_width"`
		Diagonals []struct {
			Offset int
			W64    []int
		}
	}
	if err := json.Unmarshal(source, &original); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(program, &p); err != nil {
		return nil, err
	}
	if p.Schema != "snabf.a2b.fullslot-plan.v1" || p.Source != sourceID || p.Input != original.Input || p.Output != original.Output || p.Input <= 4096 || p.Output <= 4096 || p.Input > 8192 || p.Output > 8192 || len(p.Basis.Q) != 6 || len(p.Basis.P) != 1 || p.Basis.D < 36 || p.Basis.D > 49 {
		return nil, fmt.Errorf("unsupported full-slot physical graph")
	}
	s, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 14, Q: p.Basis.Q, P: p.Basis.P, LogDefaultScale: p.Basis.D})
	if err != nil {
		return nil, err
	}
	d, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 13, Q: p.Basis.Q[:2], P: p.Basis.P, LogDefaultScale: p.Basis.D})
	if err != nil {
		return nil, err
	}
	if (p.Selected == "force-down") != (keys.Split != nil) || (p.Selected != "keep-source" && p.Selected != "force-down") {
		return nil, fmt.Errorf("wrong evaluation key roles")
	}
	id, err := keys.Commitment(s, d)
	if err != nil {
		return nil, err
	}
	if id != pin {
		return nil, fmt.Errorf("graph key commitment mismatch")
	}
	w := make(tower.Diagonals[float64])
	for _, row := range original.Diagonals {
		if w[row.Offset] != nil || len(row.W64) != 8192 {
			return nil, fmt.Errorf("invalid source weight row")
		}
		v := make([]float64, len(row.W64))
		for i, c := range row.W64 {
			if c < -64 || c > 64 {
				return nil, fmt.Errorf("invalid source weight")
			}
			v[i] = float64(c) / 64
		}
		w[row.Offset] = v
	}
	need := map[uint64]int{s.GaloisElement(1): 5}
	g := &PreparedGraph{s: s, d: d, p: p}
	if keys.Relin.LevelQ() != 3 || !splitKey(s, &keys.Relin.EvaluationKey, 3) {
		return nil, fmt.Errorf("unbound relinearization basis")
	}
	if p.Selected == "keep-source" {
		gs, e := tower.FullSlotRouteKeys(s, 2, w, &p.KeepRoute)
		if e != nil {
			return nil, e
		}
		for _, k := range gs {
			if _, exists := need[k]; !exists {
				need[k] = 2
			}
		}
	}
	gals := make([]*rlwe.GaloisKey, 0, len(need))
	for _, k := range keys.Source {
		l, ok := need[k.GaloisElement]
		if !ok || k.LevelQ() != l || k.NthRoot != s.RingQ().NthRoot() || !splitKey(s, &k.EvaluationKey, l) {
			return nil, fmt.Errorf("wrong source rotation inventory")
		}
		delete(need, k.GaloisElement)
		gals = append(gals, k.CopyNew())
	}
	if len(need) != 0 {
		return nil, fmt.Errorf("missing source rotation")
	}
	g.ev = ckks.NewEvaluator(s, rlwe.NewMemEvaluationKeySet(keys.Relin.CopyNew(), gals...))
	if p.Selected == "keep-source" {
		args := tower.Parameters{DiagonalsIndexList: slices.Sorted(maps.Keys(w)), LevelQ: 2, LevelP: 0, Scale: rlwe.NewScale(s.Q()[2]), LogDimensions: s.LogMaxDimensions()}
		g.keep = tower.NewTransformation(s, args, p.KeepRoute)
		if err = tower.Encode(ckks.NewEncoder(s), w, g.keep); err != nil {
			return nil, err
		}
	} else {
		if len(p.SplitPorts) != 2 || len(p.DownPorts) != 2 || len(p.Children) != 4 || keys.Split.Down.LevelQ() != 1 || keys.Split.Fold.LevelQ() != 1 {
			return nil, fmt.Errorf("unbound child domains or active switch basis")
		}
		for k, ports := range [][]childPort{p.SplitPorts, p.DownPorts} {
			width := p.Input
			if k == 1 {
				width = p.Output
			}
			for i, port := range ports {
				if port.Start != 4096*i || port.Used != min(4096, width-port.Start) || port.N != d.N() || port.Level != 1-k || port.Domain != "target" || !slices.Equal(port.Q, d.Q()) || !slices.Equal(port.ActiveQ, d.Q()[:2-k]) || !slices.Equal(port.P, d.P()) {
					return nil, fmt.Errorf("invalid physical child port")
				}
			}
		}
		x := FullSlotLinearPlan{InputWidth: p.Input, OutputWidth: p.Output, FoldOffsets: p.Fold.Offsets}
		for i, c := range p.Children {
			if c.Out != i/2 || c.In != i%2 {
				return nil, fmt.Errorf("noncanonical quadrant order")
			}
			x.Routes[c.Out][c.In] = c.Route
		}
		// The separately supplied graph pin above already binds the exact split pin.
		sub, e := keys.Split.Commitment(s, d, 2)
		if e != nil {
			return nil, e
		}
		g.linear, err = NewFullSlotLinear(s, d, 2, w, x, *keys.Split, sub)
		if err != nil {
			return nil, err
		}
	}
	return g, nil
}

type GraphObservation struct {
	CutSHA256, CutScale string
	Down                FullSlotLinearObservation
	Keep                tower.TransformCounts
}

func exactScale(c *rlwe.Ciphertext) string { r, _ := c.Scale.Value.Rat(nil); return r.RatString() }
func (g *PreparedGraph) Eval(x *rlwe.Ciphertext) (out []SlotBlock[*rlwe.Ciphertext], trace GraphObservation, err error) {
	// Full-size source inputs are supplied by the trusted input encoder with the
	// checked domain and zero tail. Ciphertexts do not authenticate a secret ID.
	if x == nil || x.MetaData == nil || x.Degree() != 1 || x.Level() != 5 || !x.IsBatched || !x.IsNTT || x.IsMontgomery || x.IsBitReversed || x.LogDimensions != g.s.LogMaxDimensions() || x.Scale.Mod != nil || !x.Scale.Equal(g.s.DefaultScale()) {
		return nil, trace, fmt.Errorf("wrong source ciphertext geometry/scale")
	}
	for _, poly := range x.Value {
		if poly.Level() != 5 {
			return nil, trace, fmt.Errorf("wrong input depth")
		}
		for _, v := range poly.Coeffs {
			if len(v) != g.s.N() {
				return nil, trace, fmt.Errorf("wrong input ring")
			}
		}
	}
	e := g.ev
	r, err := e.RotateNew(x, 1)
	if err != nil {
		return nil, trace, err
	}
	u, err := e.AddNew(x, r)
	if err != nil {
		return nil, trace, err
	}
	u, err = e.MulNew(u, 0.125)
	if err != nil {
		return nil, trace, err
	}
	if err = e.Rescale(u, u); err != nil {
		return nil, trace, err
	}
	v, err := e.MulNew(u, 0.25)
	if err != nil {
		return nil, trace, err
	}
	if err = e.Rescale(v, v); err != nil {
		return nil, trace, err
	}
	v, err = e.AddNew(v, 0.5)
	if err != nil {
		return nil, trace, err
	}
	e.DropLevel(u, 1)
	cut, err := e.MulRelinNew(u, v)
	if err != nil {
		return nil, trace, err
	}
	if err = e.Rescale(cut, cut); err != nil {
		return nil, trace, err
	}
	trace.CutScale = exactScale(cut)
	if cut.Level() != 2 || trace.CutScale != g.p.Keep.Cut || trace.CutScale != g.p.Down.Cut {
		return nil, trace, fmt.Errorf("live prefix differs from checked level/scale")
	}
	raw, err := cut.MarshalBinary()
	if err != nil {
		return nil, trace, err
	}
	trace.CutSHA256 = fmt.Sprintf("%x", sha256.Sum256(raw))
	if g.linear == nil {
		z, c, err := tower.NewEvaluator(e).EvaluateCounted(cut, g.keep)
		trace.Keep = c
		if err != nil {
			return nil, trace, err
		}
		if err = e.Rescale(z, z); err != nil {
			return nil, trace, err
		}
		if z.Level() != 1 || exactScale(z) != g.p.Keep.Sink {
			return nil, trace, fmt.Errorf("wrong keep sink")
		}
		return []SlotBlock[*rlwe.Ciphertext]{{0, g.p.Output, z}}, trace, nil
	}
	parts, counts, err := g.linear.Eval(cut)
	trace.Down = counts
	if err != nil {
		return nil, trace, err
	}
	if counts.Split.FoldGaloisElement != g.p.Fold.Galois || counts.Split.SlotMapID != g.p.Fold.SlotMap {
		return nil, trace, fmt.Errorf("live trace orbit mismatch")
	}
	for i, p := range parts {
		a, b := g.p.SplitPorts[i], g.p.DownPorts[i]
		split := counts.Split.Outputs[i]
		if split.N != a.N || split.Level != a.Level || split.LogicalStart != a.Start || split.UsedSlots != a.Used || split.ExactScale != a.Scale || p.Start != b.Start || p.Used != b.Used || p.Value.Level() != b.Level || p.Value.Value[0].N() != b.N || exactScale(p.Value) != b.Scale {
			return nil, trace, fmt.Errorf("live target result/scale differs from checked port")
		}
	}
	return parts[:], trace, nil
}
