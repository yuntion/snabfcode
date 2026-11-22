#pragma once
#include "GadgetNoiseContract.h"
#include <iosfwd>
#include <map>
namespace snabf::pass3 {
Solution optimize(const Graph &, const Options &);
// One level-zero edge's actual pre-solve base-width term, not a concrete Q.
double baseWidthLower(const Graph &, int edge, int defaultScale, double guard);
// Trace-only IR inputs discarded by absolute-value propagation. An arbitrary
// caller-supplied Graph has no such provenance; do not infer a signed literal
// by subtracting two rounded magnitudes.
struct IRMagnitudeOrigins {
  std::string intervalMode;
  std::map<int, double> polynomialLiterals;
};
std::ostream *rejectionTraceStream();
// The same search, with one JSON line per rejected attempt/refinement. The
// two-argument entry point writes these lines to stderr only when
// SNABF_PASS3_TRACE_REJECTIONS=1. A record concerns one candidate, not the
// entire search. Numeric fields are decimal strings (including exact Q/P).
// Optional isolation inputs describe a construction guard or the original
// sparse rows of that solve round. They do not change admission or name an IIS.
Solution optimize(const Graph &, const Options &, std::ostream &rejections);
Solution optimize(const Graph &, const Options &, std::ostream &rejections,
                  const IRMagnitudeOrigins &origins);
// A live ciphertext fixes Q, P and its scale. Re-solve the remaining integer
// gadget decisions using those actual primes; no change of basis is implicit.
Solution optimizeAtBoundary(const Graph &,const Options &,Solution physical);
}
