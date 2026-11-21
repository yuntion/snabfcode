#pragma once

#include "NativeLinalgImport.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/SourceMgr.h"

namespace snabf::native {
namespace json = llvm::json;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::Value;
using llvm::StringRef;

struct Source {
  llvm::SourceMgr &mgr;
  mlir::AsmParserState &parser;
  std::string digest;
  std::pair<unsigned, unsigned> point(Operation *op) const;
  json::Object location(Operation *op) const;
  LogicalResult reject(Operation *op, const llvm::Twine &why) const;
  LogicalResult attrs(Operation *op, llvm::ArrayRef<StringRef> allowed) const;
  LogicalResult strictFloatOp(Operation *op) const;
};

std::string typeText(mlir::Type type);
std::string floatBits(const llvm::APFloat &value);

struct Entry {
  mlir::func::FuncOp func;
  mlir::RankedTensorType vector;
  json::Object input;
};
LogicalResult inspectEntry(const Source &, mlir::ModuleOp, StringRef spec, Entry &);

// Each ID is a definition (never a printed SSA name), assigned in lexical
// order. Inner block IDs are scoped by the outer node's ID.
using ValueIDs = llvm::DenseMap<Value, std::string>;
// Invoked on the verified source operation, not on serialized receipt data.
// A later operation can still fail: callers must stage their own outputs.
using CheckedMatvec = llvm::function_ref<LogicalResult(const Source &, mlir::linalg::MatvecOp,
                                                       const ValueIDs &, StringRef resultID)>;
LogicalResult inspectWithMatvecs(StringRef bytes, StringRef name, StringRef spec,
                                 json::Object &receipt, llvm::raw_ostream &diagnostics,
                                 CheckedMatvec visit);
LogicalResult lockActivation(const Source &, mlir::linalg::GenericOp,
                             const ValueIDs &, StringRef site, json::Object &lock);
} // namespace snabf::native
