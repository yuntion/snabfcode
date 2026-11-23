#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace snabf::pass2 {

struct Interval1D {
  double lo, hi;
};

enum class StageFunction { SiluDirect, TanhQuarter, Rational, GeluDirect, ReluSignPolynomial };
enum class FitBasis { OddChebyshevRadiusNormalized, LiteralOddPowers };
enum class FitDomain { PropagatedIdealInput, RegisteredChain };
enum class CoefficientFormat { AscendingBinary64OriginalPowers };
enum class Reconstruction { MultiplyOriginalInputByOffsetPlusCore, HalfInputTimesOnePlusCore };

inline constexpr std::array<unsigned, 6> kCandidateDegrees{3, 5, 7, 9, 13, 15};

struct StageSpec {
  StageFunction function;
  std::string_view id, referenceExpression;
  // Interval containing all ideal inputs to this stage in its template.
  // In particular tanh(x/4) feeds Rational with a number in [-1,1].
  // Standalone fitting/certification can also evaluate Rational outside it.
  Interval1D templateInputEnvelope;
  std::array<unsigned, 6> degrees = kCandidateDegrees;
  FitBasis basis = FitBasis::OddChebyshevRadiusNormalized;
  FitDomain fitDomain = FitDomain::PropagatedIdealInput;
  CoefficientFormat coefficientFormat = CoefficientFormat::AscendingBinary64OriginalPowers;
};

// Both recipes carry the original x past the odd stages. The reconstruction
// enum distinguishes x*(0.5+core) from (x/2)*(1+core).
struct PolyTemplateSpec {
  std::string id;
  double templateRefError;
  std::vector<StageFunction> stages;
  Reconstruction reconstruction = Reconstruction::MultiplyOriginalInputByOffsetPlusCore;
  double reconstructionOffset = 0.5;
};

// Static, immutable storage; unknown activation names return an empty list.
const std::vector<PolyTemplateSpec>& templatesFor(std::string_view activation);
const StageSpec& stageSpec(StageFunction stage);
std::string_view stageName(StageFunction stage);

struct StagePolynomial {
  unsigned degree;
  // Ascending powers of the ORIGINAL input x, not Chebyshev coefficients.
  std::vector<double> coefficients;
};

StagePolynomial generateStage(StageFunction stage, unsigned degree,
                              Interval1D Iin);

}  // namespace snabf::pass2
