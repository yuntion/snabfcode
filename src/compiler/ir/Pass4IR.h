#pragma once
#include "Pipeline.h"
namespace snabf::ir {
void planRings(ModuleOp, int64_t minRing, unsigned rounds,
               double conversionPrice, StringRef audit);
bool isMixed(ModuleOp);
void checkRings(ModuleOp);
void checkRingSearch(ModuleOp);
DictionaryAttr ringProgram(ModuleOp);
} // namespace snabf::ir
