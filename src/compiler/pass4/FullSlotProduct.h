#pragma once
#include "mlir/IR/BuiltinOps.h"

namespace snabf::pass4 {
bool isFullSlotProduct(mlir::ModuleOp);
mlir::LogicalResult planFullSlotProduct(mlir::ModuleOp, llvm::StringRef proposal);
mlir::LogicalResult checkFullSlotProduct(mlir::ModuleOp, llvm::StringRef security, bool requireDown);
mlir::LogicalResult lowerFullSlotProduct(mlir::ModuleOp, llvm::StringRef security, llvm::StringRef output);
}
