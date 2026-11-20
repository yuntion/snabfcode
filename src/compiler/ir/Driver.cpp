#include "Pipeline.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<snabf::ir::SNABF, mlir::heir::lattigo::LattigoDialect>();
  snabf::ir::registerPasses();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "SN-ABF budget-aware compiler", registry));
}
