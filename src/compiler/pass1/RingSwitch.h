#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace snabf {
// Slot i in the incoming encoding denotes logical value i % period, with
// padding zeros after live. period is bound to the SSA input, not a cost.
struct BootContract {
  std::int64_t n, target, level, maxLevel, live, period, beta;
  std::string layout; // trace-periodic | split-repack
  bool layoutAdmitted;
  double keyVariance, down, up, prepare;
  std::vector<double> weights;
  std::vector<std::pair<double, double>>
      downstreamBins; // additional variance -> cost
  std::map<std::pair<std::int64_t, std::int64_t>, double> bootCosts;
  std::int64_t minSmallLevel = 0;
  double incomingVariance = 0;
  std::optional<double> varianceCeiling;
};
struct BootDecision {
  bool enabled = false;
  std::string reason;
  std::optional<double> rsVariance, outputVariance, downstreamPenalty, savings,
      penalty, margin;
};
BootDecision evaluateRingSwitch(const BootContract &);
} // namespace snabf
