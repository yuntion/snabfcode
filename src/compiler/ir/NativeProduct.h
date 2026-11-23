#pragma once
#include "mlir/IR/BuiltinOps.h"

namespace snabf::native {
bool isProduct(mlir::ModuleOp);
mlir::LogicalResult planProduct(mlir::ModuleOp, llvm::StringRef proposal);
mlir::LogicalResult checkProduct(mlir::ModuleOp, llvm::StringRef security);
mlir::LogicalResult lowerProduct(mlir::ModuleOp, llvm::StringRef security, llvm::StringRef output);
} // namespace snabf::native
