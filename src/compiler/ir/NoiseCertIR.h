#pragma once

#include "ExactScaleTracker.h"
#include "NoiseCertVerifier.h"
#include "Pipeline.h"

namespace snabf::ir {
// An old/unbound module has no trusted source/plaintext scale, not even for
// an integral log. No external callback can authorize a guessed binding.
noise::Topology checkedNoiseTopology(ModuleOp);
// Optional non-serializing view. A v1 log argument and the rational passed
// to NewScale are different quantities; the latter is in receipt.requested_scale.
struct V1EncodingCall { EncodingCallReceipt receipt; ExactRational log_argument; };
noise::Topology checkedNoiseTopology(ModuleOp, std::vector<V1EncodingCall> &calls);
llvm::json::Object noiseAuditJSON(const noise::CertAuditReport &);

// No analytic provider in the shipped MLIR path. Writes JSON even on a
// certificate denial and throws ContractError afterwards; never downgrades.
void auditNoisePlan(ModuleOp, StringRef outputPath, bool requireCertificate,
                    unsigned precisionBits);
void auditNoisePlan(ModuleOp, StringRef outputPath, bool requireCertificate,
                    unsigned precisionBits, double targetFailureProb, bool b2i = false);
} // namespace snabf::ir
