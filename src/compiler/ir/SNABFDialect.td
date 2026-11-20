include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"

def SNABF : Dialect {
  let name = "snabf";
  let cppNamespace = "::snabf::ir";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}
