#pragma once
#include "GadgetNoiseContract.h"
#include <iosfwd>
#include <map>
#include <optional>
#include <utility>
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
struct BoundaryRow {
  double fixed; // normalized by the unchanged epsilon, exactly as submitted
  std::map<int, double> parents, choices; // DAG index -> gain; allowed K -> increment
};
struct BoundaryRows {
  std::vector<BoundaryRow> rows;
  bool provenInfeasible = false;
};
struct BoundaryProjection {
  std::string status;
  int blockedEdge = -1, internalCapEdge = -1;
  std::vector<double> floors, gains; // decoded error; sink-error gain per crossing
  double suffix = 0, jointBudget = 0, demand = 0, deficit = 0;
};
BoundaryProjection projectBoundaryCut(const Graph &, const Options &, const BoundaryRows &,
                                      const std::vector<std::pair<int, int>> &producerDown);
Solution optimizeAtBoundary(const Graph &,const Options &,Solution physical,
                            BoundaryRows *capture = nullptr);

// A single, joint decoded-error constraint on specific source SSA results.
// It augments the normal source MILP; it never replaces its sink constraint.
struct SourceCutBound {
  std::map<int, double> weights;
  double budget;
};
struct SourceCutStats {
  unsigned candidates = 0, modelsWithCuts = 0, solverRuns = 0, solverInfeasible = 0,
           concreteCutMisses = 0;
};
std::optional<Solution> optimizeForCuts(const Graph &, const Options &,
                                      const std::vector<SourceCutBound> &, SourceCutStats &);
}
