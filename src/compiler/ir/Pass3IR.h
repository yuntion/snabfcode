#pragma once
#include "Pipeline.h"
#include "pass3/GadgetNoiseContract.h"
#include "llvm/ADT/DenseMap.h"
namespace snabf::pass3 { struct BoundaryRows; }
namespace snabf::ir {
void publishPass3(ModuleOp, const pass3::Options &, bool fixed, StringRef audit,
                  StringRef security, double sourceFraction);
void verifyPass3(ModuleOp);
DictionaryAttr boundaryParameters(ModuleOp, DictionaryAttr baseline,
                                  DictionaryAttr replay = {}, pass3::BoundaryRows *capture = nullptr);
// Feedback may propose a profile; it neither installs it nor authorizes a conversion.
DictionaryAttr boundaryFeedback(ModuleOp source, ModuleOp mixed, DictionaryAttr baseline,
                                ArrayRef<Value> sourceValues, ArrayRef<Value> mixedValues,
                                ArrayAttr uses, StringRef proposal, const pass3::BoundaryRows &initial,
                                StringRef disabledReason);
// Keys are actual SSA results, not guessed positions in the expanded arithmetic
// DAG.
struct ConcreteSSAContext {
  llvm::DenseMap<Value, DictionaryAttr> values;
  ArrayAttr keyDomains;
};
ConcreteSSAContext concreteSSAContext(ModuleOp,
                                      DictionaryAttr verifiedParameters);
void verifySourceScreen(ModuleOp, DictionaryAttr);
void bindPass3(DictionaryAttr, llvm::json::Array &);
} // namespace snabf::ir
