#pragma once
#include "Pipeline.h"
#include "RouteSelection.h"
namespace snabf::ir {
struct Catalog {
  SmallVector<LintransOp> ops;
  std::vector<LinearRegion> regions;
  std::map<ScopedGaloisKey, uint64_t> sizes;
  std::vector<RouteDepthCut> layoutCuts;
  SmallVector<DictionaryAttr> contexts;
  double eta;
  explicit Catalog(ModuleOp m);
};
DictionaryAttr routeAttr(Builder &, const SubgroupRoute &);
ArrayAttr alternatives(Builder &, const LinearRegion &);
void publishRoutes(const Catalog &, Builder &);
DictionaryAttr chargeRecord(Builder &, const RoutingPlan &, size_t site,
                            double eta);
unsigned evaluateBootstraps(ModuleOp, bool checkOnly);
DictionaryAttr executableBootstrap(ModuleOp);
} // namespace snabf::ir
