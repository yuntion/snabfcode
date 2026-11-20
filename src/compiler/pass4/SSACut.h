#pragma once
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <stdexcept>

namespace snabf::pass4 {
struct CrossingUse {
  mlir::OpOperand *use;
  bool reachesResidualJoin;
};
struct BlockCut {
  llvm::DenseMap<mlir::Operation *, unsigned> order;
  llvm::SmallVector<CrossingUse> uses;
};

// All uses crossing the program point immediately before 'first'. The forward
// and reverse sweeps classify only joins downstream of 'main', not any add.
inline BlockCut enumerateCut(mlir::Operation *first, mlir::Operation *main,
                             llvm::function_ref<bool(mlir::Type)> encrypted,
                             llvm::function_ref<bool(mlir::Operation *)> join) {
  using namespace mlir;
  auto *block = first->getBlock();
  if (!block || main->getBlock() != block || !block->getParent()->hasOneBlock())
    throw std::invalid_argument("unsupported_multiblock_encrypted_cut");
  BlockCut cut;
  for (auto [i, op] : llvm::enumerate(*block))
    cut.order[&op] = i;
  auto pos = [&](Operation *o) { return cut.order.find(o)->second; };
  llvm::DenseSet<Operation *> fromMain{main}, reachesJoin;
  for (Operation &op : *block) {
    if (pos(&op) < pos(first))
      continue;
    if (op.getNumRegions())
      throw std::invalid_argument("unsupported_nested_suffix_cut");
    for (Value v : op.getOperands())
      if (encrypted(v.getType()) && fromMain.contains(v.getDefiningOp()))
        fromMain.insert(&op);
    if (fromMain.contains(&op) && join(&op))
      reachesJoin.insert(&op);
  }
  for (Operation &op : llvm::reverse(*block)) {
    if (pos(&op) < pos(first))
      break;
    if (!reachesJoin.contains(&op))
      continue;
    for (Value v : op.getOperands()) {
      auto *def = v.getDefiningOp();
      if (encrypted(v.getType()) && def && def->getBlock() == block &&
          pos(def) >= pos(first))
        reachesJoin.insert(def);
    }
  }
  for (Operation &op : *block) {
    if (pos(&op) < pos(first))
      continue;
    for (auto &u : op.getOpOperands()) {
      Value v = u.get();
      auto *def = v.getDefiningOp();
      if (def ? def->getBlock() != block
              : cast<BlockArgument>(v).getOwner() != block)
        throw std::invalid_argument("unsupported_nonlocal_cut_operand");
      if (!def || pos(def) < pos(first))
        cut.uses.push_back({&u, reachesJoin.contains(&op)});
    }
  }
  return cut;
}
} // namespace snabf::pass4
