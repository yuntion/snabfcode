#include "NativeLinalgImport.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  constexpr llvm::StringLiteral src = "module {func.func @id(%x: tensor<4xf32>) -> tensor<4xf32> {return %x : tensor<4xf32>}}";
  constexpr llvm::StringLiteral spec = R"({"schema":"snabf.native-input-spec.v1","inputs":[{"arg_index":0,"func_symbol":"id","classification":"secret","source_id":"s"}]})";
  llvm::json::Object receipt;
  std::string diagnostics;
  llvm::raw_string_ostream os(diagnostics);
  auto run = [&](llvm::StringRef bytes, llvm::StringRef name, llvm::StringRef ports) {
    return snabf::native::inspectNativeLinalg(bytes, name, ports, receipt, os);
  };
  if (mlir::failed(run(src, "first.mlir", spec))) { llvm::errs() << diagnostics; return 1; }
  auto text = [&] { return llvm::formatv("{0}", llvm::json::Value(llvm::json::Object(receipt))).str(); };
  const auto first = text();
  if (mlir::failed(run(src, "renamed.mlir", spec)) || text()!=first) return 2;
  auto srcSlice = src.str()+"unrelated-bytes", specSlice = spec.str()+"unrelated-bytes";
  if (mlir::failed(run(llvm::StringRef(srcSlice).take_front(src.size()), "slice.mlir",
                       llvm::StringRef(specSlice).take_front(spec.size()))) || text()!=first) return 5;
  if (mlir::succeeded(run(src, "bad-port.mlir", "{}")) || text()!=first || diagnostics.find("bad-port.mlir:1:")==std::string::npos) return 3;
  auto nul = src.str() + '\0' + "trailing";
  if (mlir::succeeded(run(nul, "nul.mlir", spec)) || text()!=first || diagnostics.find("embedded NUL")==std::string::npos) return 4;
  llvm::outs() << "API: path-independent witness, failed calls preserve receipt, embedded NUL refused\n";
  return 0;
}
