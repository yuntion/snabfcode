#include "pass4/SSACut.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <iostream>

using namespace mlir;

// Exercise the same block cut used by Pass4IR with actual MLIR Values and
// OpOperands. Numeric types here merely distinguish encrypted and public ports.
static void check(unsigned shape, unsigned expectedValues,
                  unsigned expectedUses, unsigned expectedBypasses) {
  MLIRContext ctx;
  ctx.allowUnregisteredDialects();
  OpBuilder b(&ctx);
  auto loc = b.getUnknownLoc();
  OwningOpRef<ModuleOp> m(ModuleOp::create(loc));
  b.setInsertionPointToStart(m->getBody());
  auto op = [&](StringRef name, ValueRange args, Type t) {
    OperationState state(loc, name);
    state.addOperands(args);
    state.addTypes(t);
    return b.create(state);
  };
  auto ct = b.getI32Type();
  Value x = op("test.source", {}, ct)->getResult(0);
  Value y = op("test.poly", x, ct)->getResult(0);
  Value table = op("test.table", {}, b.getF64Type())->getResult(0);
  if (shape == 1)
    y = op("test.add", {x, y}, ct)->getResult(0); // pre-merged residual
  auto main = op("test.lin", {y, table}, ct);
  Value output = main->getResult(0);
  if (shape == 2)
    output = op("test.add", {output, x}, ct)->getResult(0); // distinct values
  if (shape == 3)
    output =
        op("test.add", {output, y}, ct)->getResult(0); // same value, two uses
  if (shape == 4)
    output = op("test.add", {output, op("test.align", x, ct)->getResult(0)}, ct)
                 ->getResult(0);
  if (shape == 5)
    op("test.add", {x, x}, ct); // foreign add, not downstream of main
  op("test.sink", output, ct);
  auto cut = snabf::pass4::enumerateCut(
      main, main, [&](Type t) { return t == ct; },
      [&](Operation *o) { return o->getName().getStringRef() == "test.add"; });
  llvm::DenseSet<Value> live;
  unsigned uses = 0, publics = 0, bypasses = 0;
  for (auto item : cut.uses) {
    Value value = item.use->get();
    if (value == table) {
      ++publics;
      continue;
    }
    ++uses;
    live.insert(value);
    if (item.use->getOwner() != main && item.reachesResidualJoin)
      ++bypasses;
  }
  if (publics != 1 || live.size() != expectedValues || uses != expectedUses ||
      bypasses != expectedBypasses)
    throw std::runtime_error("incorrect SSA cut for shape " +
                             std::to_string(shape));
}
int main() {
  check(0, 1, 1, 0);
  check(1, 1, 1, 0);
  check(2, 2, 2, 1);
  check(3, 1, 2, 1);
  check(4, 2, 2, 1);
  check(5, 2, 3, 0);
  std::cout << "6 MLIR SSA cut cases passed\n";
}
