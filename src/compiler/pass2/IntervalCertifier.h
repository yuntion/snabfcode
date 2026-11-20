#pragma once

#include "PolyTemplates.h"

#include <span>

namespace snabf::pass2 {

struct StageCertificate {
  Interval1D Iin, Ain, J, Iout, Aout;
  double epsilon, L;
};

// Independently certifies the supplied binary64 coefficients as exact reals.
// Iout encloses g(Iin); Aout encloses p(Ain); J = hull(Iin, Ain).
// epsilon >= sup_Iin |p-g|; L >= sup_J |p'|. No CKKS evaluation error is
// included. Invalid or unrepresentable inputs/bounds fail by exception.
StageCertificate certifyStage(StageFunction stage, const StagePolynomial& p,
                              Interval1D Iin, Interval1D Ain);

struct ReluReconstructionCertificate {
  Interval1D input;
  double leftEpsilon, rightEpsilon, epsilon;  // [lo,0], [0,hi], max of both.
};

// g = chain.back() o ... o chain.front(); coefficients are exact binary64
// values in ascending original powers. On the finite closed input containing
// zero, epsilon >= sup |x*(1+g(x))/2 - max(x,0)|. The chain must be nonempty;
// each stage must be odd (zero is allowed), with degree <= kCandidateDegrees.back().
// Invalid data or a nonrepresentable bound throws. No CKKS error is included.
ReluReconstructionCertificate certifyReluReconstruction(
    std::span<const StagePolynomial> chain, Interval1D input);

}  // namespace snabf::pass2
