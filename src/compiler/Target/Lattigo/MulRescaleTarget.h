#pragma once
#include "llvm/Support/CommandLine.h"

namespace snabf::ir::lattigo_target::detail {
// Target selection, never an assertion read from checked IR or its audit.
inline llvm::cl::opt<bool> mulRescaleTarget("snabf-b2-mul-rescale", llvm::cl::init(false),
    llvm::cl::desc("Target the guarded single-ring DirectGo EvalFresh mul/rescale contract"));
}
