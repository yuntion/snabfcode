#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace snabf::ir { void registerPasses(); }
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  snabf::ir::registerPasses();
  return mlir::failed(mlir::MlirOptMain(argc, argv, "SN-ABF native-linalg compiler", registry));
}
