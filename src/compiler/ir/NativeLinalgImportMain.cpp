#include "NativeLinalgImport.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
namespace {
constexpr StringLiteral usage = "usage: snabf-native-import SOURCE.mlir --input-spec SPEC.json [-o RECEIPT.json]\n";
}

int main(int argc, char **argv) {
  if (argc == 2 && StringRef(argv[1]) == "--help") { outs() << usage; return 0; }
  StringRef source, spec, output;
  for (int i = 1; i < argc; ++i) {
    StringRef arg(argv[i]);
    if ((arg == "--input-spec" || arg == "-o") && i+1 < argc) {
      auto &dest = arg == "-o" ? output : spec;
      if (dest.empty() && !StringRef(argv[i+1]).starts_with("-")) { dest = argv[++i]; continue; }
    } else if (!arg.starts_with("-") && source.empty()) { source = arg; continue; }
    errs() << usage;
    return 2;
  }
  if (source.empty() || spec.empty()) { errs() << usage; return 2; }
  auto src = MemoryBuffer::getFile(source, /*IsText=*/false);
  if (!src) { errs() << source << ": " << src.getError().message() << '\n'; return 1; }
  auto in = MemoryBuffer::getFile(spec, /*IsText=*/false);
  if (!in) { errs() << spec << ": " << in.getError().message() << '\n'; return 1; }
  json::Object receipt;
  if (mlir::failed(snabf::native::inspectNativeLinalg((*src)->getBuffer(), source, (*in)->getBuffer(), receipt, errs()))) return 1;
  auto text = formatv("{0:2}\n", json::Value(std::move(receipt))).str();
  if (output.empty()) { outs() << text; return 0; }
  // Only open a fresh destination after every check passes. In particular a
  // misspelled output path cannot truncate either immutable input or a receipt.
  std::error_code ec;
  raw_fd_ostream file(output, ec, sys::fs::CD_CreateNew, sys::fs::FA_Write, sys::fs::OF_None);
  if (ec) { errs() << output << ": " << ec.message() << '\n'; return 1; }
  file << text;
  file.close();
  if (file.has_error()) {
    errs() << output << ": " << file.error().message() << '\n';
    file.clear_error();
    sys::fs::remove(output);
    return 1;
  }
  return 0;
}
