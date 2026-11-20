package epochs

import "github.com/tuneinsight/lattigo/v6/schemes/ckks"

// SourceEvaluation returns the live source-domain evaluation context. It does
// not generate new keys. Use it serially with Eval (the evaluator has scratch
// buffers); Decode accepts the target domain, which may differ from the source.
func (r *Runtime) SourceEvaluation() (*ckks.Evaluator, ckks.Parameters, *ckks.Encoder) {
	f := r.families[r.Program.Source]
	return f.eval, f.p, f.enc
}
