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
  double leftEpsilon, rightEpsilon, epsilon;
};
// g is the composition in evaluation order. Compare x*(1+g(x))/2 to
// max(x,0) on BOTH closed half intervals; no CKKS noise is included.
ReluReconstructionCertificate certifyReluReconstruction(
    std::span<const StagePolynomial> chain, Interval1D input);

}  // namespace snabf::pass2
