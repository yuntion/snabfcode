#pragma once

#include "NoiseCertVerifier.h"
#include "Pipeline.h"

namespace snabf::ir {
// An old/unbound module has no trusted source/plaintext scale, not even for
// an integral log. No external callback can authorize a guessed binding.
noise::Topology checkedNoiseTopology(ModuleOp);
llvm::json::Object noiseAuditJSON(const noise::CertAuditReport &);

// No analytic provider in the shipped MLIR path. Writes JSON even on a
// certificate denial and throws ContractError afterwards; never downgrades.
void auditNoisePlan(ModuleOp, StringRef outputPath, bool requireCertificate,
                    unsigned precisionBits);
void auditNoisePlan(ModuleOp, StringRef outputPath, bool requireCertificate,
                    unsigned precisionBits, double targetFailureProb);
} // namespace snabf::ir
