#pragma once

#include "PolyTemplates.h"

namespace snabf::pass2 {

struct NonlinearOpSpec {
  std::string activation;
  // Declaration order is the existing search/serialization order.
  std::vector<PolyTemplateSpec> templates;
};

// Immutable catalog of stage recipes. Lookup does not certify a source
// witness, a new mathematical identity, or a ciphertext implementation.
class NonlinearOpRegistry {
 public:
  explicit NonlinearOpRegistry(std::vector<NonlinearOpSpec> ops);
  const NonlinearOpSpec* find(std::string_view activation) const;
  const PolyTemplateSpec& requireTemplate(std::string_view activation, std::string_view id) const;

 private:
  const std::vector<NonlinearOpSpec> ops_;
};

const NonlinearOpRegistry& builtinNonlinearOps();

}  // namespace snabf::pass2
