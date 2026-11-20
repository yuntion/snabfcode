#pragma once
#include "Pipeline.h"

namespace snabf::ir::lattigo_target {
// Check, lower and emit are separate MLIR invocations; a marker is never a
// replacement for replay. Only these module-level APIs can publish artifacts.
void checkAndSeal(ModuleOp);
void lowerDirect(ModuleOp);
void emitDirect(ModuleOp, StringRef goPath, StringRef auditPath);

namespace detail {
std::string renderGo(DictionaryAttr verifiedProgram, StringRef digest);
}
} // namespace snabf::ir::lattigo_target
