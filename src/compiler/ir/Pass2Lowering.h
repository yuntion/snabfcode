#pragma once
#include "Pipeline.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <map>

namespace snabf::ir {
// State for the single outlined evaluator supported by the HEIR importer.
class Pass2Lowering {
 public:
  Pass2Lowering(ModuleOp module, ProfileOp profile, IRRewriter &rw);
  bool active() const { return bool(eval); }
  void bindSource(SourceOp op);
  Value emit(PolyEvalOp poly);
  Type nativeResult(Operation *op, NamedAttrList &properties);
  void bindSink(SinkOp op);
  void finish();

 private:
  ModuleOp module;
  ProfileOp profile;
  IRRewriter &rw;
  func::FuncOp eval, prep;
  Value readStorage, writeStorage;
  Type oldSink, newSink;
  std::map<int64_t, Operation *> stores;
  std::map<int64_t, int64_t> boundLevels;
  Value coefficient(Location loc, double value, int64_t level, int64_t logScale,
                    double physicalScale, Type ciphertext);
  void bindTable(Value table, int64_t level, bool prepared);
};

// In emitted-function order. Only the scale operand of marked plaintext
// encodings needs a backend binding; the polynomial itself is HEIR SSA.
llvm::json::Array pass2ScaleBindings(ModuleOp m, int64_t expected);
}  // namespace snabf::ir
