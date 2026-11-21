#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "mlir/Support/LogicalResult.h"

namespace snabf::native {
// Parse the exact supplied byte buffers and verify the N1 source subset.
// On failure write a diagnostic and leave receipt untouched. No lowering,
// process execution or filesystem access is part of this interface.
mlir::LogicalResult inspectNativeLinalg(llvm::StringRef sourceBytes,
                                      llvm::StringRef sourceName,
                                      llvm::StringRef inputSpecBytes,
                                      llvm::json::Object &receipt,
                                      llvm::raw_ostream &diagnostics);
} // namespace snabf::native
