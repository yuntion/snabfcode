#pragma once

#include "PolyTemplates.h"

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

}  // namespace snabf::pass2
