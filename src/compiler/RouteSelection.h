#ifndef SNABF_ROUTE_SELECTION_H
#define SNABF_ROUTE_SELECTION_H

#include "pass1/KeyReuseModel.h"

#include <cstddef>
#include <utility>

namespace snabf {

// A budget witness contains linear sites on one real SSA path to a sink.
struct RouteDepthCut { std::vector<std::size_t> sites; unsigned limit; };

RoutingPlan selectSubgroupRoutes(
    const std::vector<LinearRegion>& regions,
    const std::map<ScopedGaloisKey, std::uint64_t>& keyBytes,
    double costPerKeyByte, const std::vector<RouteDepthCut>& cuts = {});

}  // namespace snabf

#endif
