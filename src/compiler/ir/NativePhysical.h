#pragma once
#include "NativeArithmetic.h"
#include "NativeScaleLower.h"

namespace snabf::native {
// Materialize the single-P k=0 engineering schedule. Bound arguments are in
// decoded units, not log2 tolerances. Does not confer security approval.
llvm::json::Object lowerNativePhysical(const NativeArithmeticProgram &, const pass3::Solution &,
                                       const pass3::Options &, const ExactRational &localJoinBudget = 0);
} // namespace snabf::native
