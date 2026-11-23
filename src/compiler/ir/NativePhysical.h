#pragma once
#include "NativeArithmetic.h"
#include "NativeScaleLower.h"

namespace snabf::native {
// The same ordered source, odd stages, bypass, DAG and both proofs are replayed
// by the checker and lowering, not accepted as a certificate from the caller.
llvm::json::Object nativeReluWitness(const NativeArithmeticProgram &);
void checkNativeReluWitness(const NativeArithmeticProgram &,const llvm::json::Object *);
// k=0 key-switch term in encoded units (rounding/rescale are separate).
ExactRational nativeBlockSwitchError(const ScaleDomain &, int activeLevel);
// Materialize a k=0 engineering schedule. Bound arguments are in
// decoded units, not log2 tolerances. Does not confer security approval.
llvm::json::Object lowerNativePhysical(const NativeArithmeticProgram &, const pass3::Solution &,
                                       const pass3::Options &, const ExactRational &localJoinBudget = 0,
                                       llvm::json::Object *reluAudit = nullptr, pass3::Solution *verified = nullptr);
} // namespace snabf::native
