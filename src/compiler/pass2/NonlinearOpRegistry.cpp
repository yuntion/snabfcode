#include "NonlinearOpRegistry.h"

#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace snabf::pass2 {

NonlinearOpRegistry::NonlinearOpRegistry(std::vector<NonlinearOpSpec> ops) : ops_(std::move(ops)) {
  std::set<std::string_view> activations, ids;
  for (const auto& op : ops_) {
    if (op.activation.empty() || !activations.insert(op.activation).second || op.templates.empty())
      throw std::invalid_argument("duplicate or empty nonlinear activation registration");
    for (const auto& spec : op.templates) {
      if (spec.id.empty() || !ids.insert(spec.id).second)
        throw std::invalid_argument("duplicate or empty nonlinear template ID");
      if (spec.stages.empty() || !std::isfinite(spec.templateRefError) || spec.templateRefError<0 ||
          !std::isfinite(spec.reconstructionOffset) ||
          spec.reconstruction!=Reconstruction::MultiplyOriginalInputByOffsetPlusCore)
        throw std::invalid_argument("invalid nonlinear template recipe");
      for (auto stage : spec.stages) (void)stageSpec(stage);
    }
  }
}

const NonlinearOpSpec* NonlinearOpRegistry::find(std::string_view activation) const {
  for (const auto& op : ops_) if (op.activation==activation) return &op;
  return nullptr;
}

const PolyTemplateSpec& NonlinearOpRegistry::requireTemplate(std::string_view activation, std::string_view id) const {
  if (const auto* op=find(activation)) {
    for (const auto& spec : op->templates) if (spec.id==id) return spec;
  }
  throw std::invalid_argument("unknown nonlinear activation/template binding");
}

const NonlinearOpRegistry& builtinNonlinearOps() {
  using F=StageFunction;
  // tanh(2z)/2 = tanh(z)/(1+tanh(z)^2); Phi(x)-1/2 = erf(x/sqrt(2))/2.
  static const NonlinearOpRegistry registry({
      {"silu", {{"silu_direct",0,{F::SiluDirect}},
                {"silu_tanh_rational",0,{F::TanhQuarter,F::Rational}}}},
      {"gelu", {{"gelu_direct",0,{F::GeluDirect}}}}});
  return registry;
}

}  // namespace snabf::pass2
