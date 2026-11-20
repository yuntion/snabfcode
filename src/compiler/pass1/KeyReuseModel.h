#pragma once
#include "SubgroupTower.h"
#include <cstddef>
#include <string>
#include <utility>

namespace snabf {
using ScopedGaloisKey = std::pair<std::string, std::uint64_t>;
using KeySizes = std::map<ScopedGaloisKey, std::uint64_t>;
struct LinearRegion {
  std::string keyScope; // Same secret/configuration, N and Q/P basis.
  double hoistedStepCost, giantStepCost;
  std::vector<SubgroupRoute> candidates;
  bool densePricing = false;
};
struct SiteCharge {
  double hoistedWork = 0, giantWork = 0, amortizedBytes = 0;
};
struct RoutingPlan {
  std::vector<std::size_t> selected;
  std::map<ScopedGaloisKey, std::size_t> referenceCounts;
  std::vector<SiteCharge> charges;
  std::uint64_t keyBytes = 0;
  double objective = 0;
};
SiteCharge operationCharge(const LinearRegion &, const SubgroupRoute &);
// Score any fixed assignment without calling the optimizer. Independent and
// joint arms pay for the SAME physical union; work is charged per invocation.
RoutingPlan scoreSelection(const std::vector<LinearRegion> &, const KeySizes &,
                           double eta, const std::vector<std::size_t> &ids);
} // namespace snabf
