#pragma once

#include "ir/ExactScaleTracker.h"
#include "pass3/ParameterMILP.h"
#include <array>

namespace snabf::pass4 {
// Two full-capacity standard CKKS rings, one padded input/output vector.
// Original diagonal k has coefficient W[row,(row+k) mod (2*S)].
struct FullSlotWeights {
  int inputWidth, outputWidth;
  std::map<int, std::vector<double>> diagonals;
};
struct FullSlotShape {
  ExactRational norm;
  int sourceTerms, maxChildTerms;
  std::array<std::array<std::vector<int>,2>,2> childOffsets;
};
FullSlotShape analyzeFullSlot(const FullSlotWeights &, int childSlots);
pass3::Graph neighborSiluPrefix();
struct FullSlotMargin {
  ExactRational cutScale, linearInputScale, sinkScale, cutError, fresh, gain, budget, total, downKeyCharge;
  bool headroom;
};
// Bounds in decoded units under the engineering primitive allowances. No samples,
// observed errors, independence assumptions or probability certificate.
FullSlotMargin fullSlotMargin(const FullSlotShape &, const pass3::Solution &, bool split);
struct FullSlotFeedback {
  pass3::Solution initial, basis;
  FullSlotMargin oldKeep, oldDown, keep, down;
  pass3::SourceCutStats stats;
  double submittedBudget;
  int marginBits;
  std::vector<std::string> rejected;
};
FullSlotFeedback solveFullSlot(const FullSlotShape &, int sourceCap);
} // namespace snabf::pass4
