#include "KeyReuseModel.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace snabf {
SiteCharge operationCharge(const LinearRegion &r, const SubgroupRoute &c) {
  auto h = c.splits.front();
  return {r.hoistedStepCost *
              (r.densePricing ? h.order - 1 : c.babyRotations.size()),
          r.giantStepCost *
              (r.densePricing ? h.index - 1 : c.giantRotations.size()),
          0};
}
RoutingPlan scoreSelection(const std::vector<LinearRegion> &regions,
                           const KeySizes &sizes, double eta,
                           const std::vector<std::size_t> &ids) {
  if (ids.size() != regions.size() || !std::isfinite(eta) || eta < 0)
    throw std::invalid_argument("invalid route assignment or key price");
  RoutingPlan p;
  p.selected = ids;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto &r = regions[i];
    const auto &c = r.candidates.at(ids[i]);
    p.charges.push_back(operationCharge(r, c));
    p.objective += p.charges.back().hoistedWork + p.charges.back().giantWork;
    for (auto k : c.galoisKeys)
      ++p.referenceCounts[{r.keyScope, k}];
  }
  for (const auto &[key, refs] : p.referenceCounts) {
    auto bytes = sizes.at(key);
    if (!bytes ||
        bytes > std::numeric_limits<std::uint64_t>::max() - p.keyBytes)
      throw std::overflow_error("invalid total Galois key storage");
    p.keyBytes += bytes;
  }
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto &r = regions[i];
    for (auto g : r.candidates.at(ids[i]).galoisKeys) {
      ScopedGaloisKey k{r.keyScope, g};
      p.charges[i].amortizedBytes +=
          double(sizes.at(k)) / p.referenceCounts.at(k);
    }
  }
  p.objective += eta * p.keyBytes;
  return p;
}
} // namespace snabf
