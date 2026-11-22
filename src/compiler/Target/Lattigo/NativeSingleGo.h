#pragma once
#include "NativePhysical.h"
namespace snabf::native {
llvm::StringRef nativeSingleRuntimeSource();
// Internal N1 engineering backend: source-bound planned schedule only. The
// release CLI remains closed until all independent gates are complete.
void checkNativeSingle(const NativeArithmeticProgram &, const pass3::Solution &,const pass3::Options &,
                       const ExactRational &joinBudget, llvm::StringRef plan, llvm::StringRef evidence);
std::string emitNativeSingleGo(const NativeArithmeticProgram &, const pass3::Solution &,const pass3::Options &,
                       const ExactRational &joinBudget, llvm::StringRef plan, llvm::StringRef evidence);
}
