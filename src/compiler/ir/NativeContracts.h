#pragma once
#include "IR.h"

namespace snabf::ir {
// N0 understands the wire schema; no production pass may consume it yet.
mlir::LogicalResult rejectNativeExecution(mlir::ModuleOp module);
mlir::LogicalResult verifyNativeLintrans(LintransOp op);
} // namespace snabf::ir
