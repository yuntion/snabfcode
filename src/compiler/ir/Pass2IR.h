#pragma once
#include "Pipeline.h"
#include "llvm/ADT/DenseMap.h"
#include <utility>

namespace snabf::ir {
void markActivationSources(ModuleOp);
LogicalResult verifyStaticSource(ModuleOp);
void liftActivationRegions(ModuleOp);
void publishPolynomials(ModuleOp, int64_t depth, double epsilon,
                        int64_t minOutputLevel, double maxOutputError, StringRef family);
void refreshPolynomials(ModuleOp, bool verify);
// Empty in legacy mode. Otherwise every entry is the hull across logical
// slots of both ideal and budget-admitted real polynomial trajectories. Use
// .find(poly.getInput()) while publishing/rechecking Pareto frontiers.
llvm::DenseMap<Value, std::pair<double,double>> affinePlanningDomains(ModuleOp);
double linearNorm(DenseFPElementsAttr weights, bool matrix);
double upperProduct(double,double);
double upperSum(double,double);
DictionaryAttr pass2Manifest(ModuleOp);
DictionaryAttr transferRecord(PolyEvalOp, DictionaryAttr candidate, int64_t inputLevel);
llvm::json::Object jsonPass2(DictionaryAttr);
}
