#pragma once
#include "Pipeline.h"

namespace snabf::ir {
void markActivationSources(ModuleOp);
void liftActivationRegions(ModuleOp);
void publishPolynomials(ModuleOp, int64_t depth, double epsilon,
                        int64_t minOutputLevel, double maxOutputError, StringRef family);
void refreshPolynomials(ModuleOp, bool verify);
double linearNorm(DenseFPElementsAttr weights, bool matrix);
double upperProduct(double,double);
double upperSum(double,double);
DictionaryAttr pass2Manifest(ModuleOp);
DictionaryAttr transferRecord(PolyEvalOp, DictionaryAttr candidate, int64_t inputLevel);
llvm::json::Object jsonPass2(DictionaryAttr);
}
