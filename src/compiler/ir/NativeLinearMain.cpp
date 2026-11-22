#include "NativeLinearRoute.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
namespace {
constexpr StringLiteral usage = "usage: snabf-native-linear SOURCE.mlir --input-spec SPEC.json "
                                "--log-n LOG_N --encoded-slots SLOTS\n";
}

int main(int argc, char **argv) {
  if (argc == 2 && StringRef(argv[1]) == "--help") { outs() << usage; return 0; }
  StringRef source, spec;
  std::optional<unsigned> logN, slots;
  for (int i = 1; i < argc; ++i) {
    StringRef arg(argv[i]);
    if (!arg.starts_with("-") && source.empty()) { source = arg; continue; }
    if (i + 1 < argc && !StringRef(argv[i + 1]).starts_with("-")) {
      if (arg == "--input-spec" && spec.empty()) { spec = argv[++i]; continue; }
      if (arg == "--log-n" || arg == "--encoded-slots") {
        auto &dest = arg == "--log-n" ? logN : slots;
        unsigned v;
        if (!dest && !StringRef(argv[i + 1]).getAsInteger(10, v)) { dest = v; ++i; continue; }
      }
    }
    errs() << usage;
    return 2;
  }
  if (source.empty() || spec.empty() || !logN || !slots) { errs() << usage; return 2; }
  auto src = MemoryBuffer::getFile(source, /*IsText=*/false);
  if (!src) { errs() << source << ": " << src.getError().message() << '\n'; return 1; }
  auto in = MemoryBuffer::getFile(spec, /*IsText=*/false);
  if (!in) { errs() << spec << ": " << in.getError().message() << '\n'; return 1; }
  snabf::native::NativeLinearProgram p;
  if (mlir::failed(snabf::native::deriveNativeLinear((*src)->getBuffer(), source, (*in)->getBuffer(),
                                                     {*logN, *slots}, p, errs()))) return 1;
  auto receipt = snabf::native::takeNativeLinearReceipt(std::move(p));
  if (mlir::failed(receipt)) { errs() << "incomplete native linear program\n"; return 1; }
  outs() << formatv("{0}\n", json::Value(std::move(*receipt)));
  return 0;
}
