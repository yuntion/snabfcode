#pragma once
#include "NativePhysical.h"
namespace snabf::native {
llvm::StringRef nativeSingleRuntimeSource();
// Source-bound single-ring backend shared by the product MLIR passes and
// isolated engineering probes. The caller supplies trusted estimator evidence.
void checkNativeSingle(const NativeArithmeticProgram &, const pass3::Solution &,const pass3::Options &,
                       const ExactRational &joinBudget, llvm::StringRef plan, llvm::StringRef evidence);
std::string emitNativeSingleGo(const NativeArithmeticProgram &, const pass3::Solution &,const pass3::Options &,
                       const ExactRational &joinBudget, llvm::StringRef plan, llvm::StringRef evidence);
}
