#pragma once
#include "mlir/IR/BuiltinOps.h"

namespace snabf::ir::lattigo_target {
// After concrete plan verification, check any existing seal before HEIR
// rewrites the original SSA. Does not seal, lower, or select a backend.
void verifySealedInputForHEIR(mlir::ModuleOp);

// Only planRings may retire emission artifacts, on its private input clone,
// after replaying any persisted ring policy and before changing the graph.
void invalidateForExplicitPass4Replan(mlir::ModuleOp);
} // namespace snabf::ir::lattigo_target
