#pragma once
#include "GadgetNoiseContract.h"
#include <iosfwd>
namespace snabf::pass3 {
Solution optimize(const Graph &, const Options &);
// The same search, with one JSON line per rejected attempt/refinement. The
// two-argument entry point writes these lines to stderr only when
// SNABF_PASS3_TRACE_REJECTIONS=1. A record concerns one candidate, not the
// entire search. Numeric fields are decimal strings (including exact Q/P).
Solution optimize(const Graph &, const Options &, std::ostream &rejections);
// A live ciphertext fixes Q, P and its scale. Re-solve the remaining integer
// gadget decisions using those actual primes; no change of basis is implicit.
Solution optimizeAtBoundary(const Graph &,const Options &,Solution physical);
}
