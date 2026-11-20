#pragma once

#include "NoiseCertVerifier.h"
#include "Pipeline.h"

namespace snabf::ir {
// An optional trusted binding to the ACTUAL backend-generated binary64
// plaintext scale for this module, instruction ID and emitted Go LogScale.
// Not a predicted scale or a value read out of a planner noise record.
using BoundEncodingScale = std::function<std::optional<double>(ModuleOp, StringRef, double)>;

// Rechecks the concrete plan and binds the expanded primitives to actual SSA
// sinks. For fractional Go math.Exp2 inputs no host-libm guess is accepted.
noise::Topology checkedNoiseTopology(ModuleOp, const BoundEncodingScale & = {});
llvm::json::Object noiseAuditJSON(const noise::CertAuditReport &);

// No analytic provider in the shipped MLIR path. Writes JSON even on a
// certificate denial and throws ContractError afterwards; never downgrades.
void auditNoisePlan(ModuleOp, StringRef outputPath, bool requireCertificate,
                    unsigned precisionBits);
void auditNoisePlan(ModuleOp, StringRef outputPath, bool requireCertificate,
                    unsigned precisionBits, double targetFailureProb);
} // namespace snabf::ir
