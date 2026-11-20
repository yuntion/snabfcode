#include "pass2/CompositePolyDP.h"
#include "pass2/NonlinearOpRegistry.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace snabf::pass2;
using F=StageFunction;

void expect(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class T> void rejects(T op, const char* why) {
  try { op(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error(why);
}

void run() {
  const auto& registry=builtinNonlinearOps();
  const auto& silu=templatesFor("silu"); const auto& gelu=templatesFor("gelu");
  expect(silu.size()==2 && gelu.size()==1 && &registry.requireTemplate("silu","silu_direct")==&silu[0] &&
         &registry.requireTemplate("silu","silu_tanh_rational")==&silu[1] &&
         &registry.requireTemplate("gelu","gelu_direct")==&gelu[0],"lookup/order differs from baseline");
  expect(silu[0].stages==std::vector{F::SiluDirect} && silu[1].stages==std::vector{F::TanhQuarter,F::Rational} &&
         gelu[0].stages==std::vector{F::GeluDirect},"registered stage chains changed");
  for (auto spec:{&silu[0],&silu[1],&gelu[0]})
    expect(spec->templateRefError==0 && spec->reconstructionOffset==0.5 &&
           spec->reconstruction==Reconstruction::MultiplyOriginalInputByOffsetPlusCore,"reconstruction changed");

  rejects([&]{registry.requireTemplate("silu","gelu_direct");},"cross-activation template accepted");
  rejects([&]{registry.requireTemplate("gelu","nonexistent");},"unknown template accepted");
  expect(!registry.find("relu") && templatesFor("relu").empty(),"uncertified ReLU registered");
  rejects([]{synthesizePolynomials("relu",{-1,1});},"actual DP admitted unsupported ReLU");
  NonlinearOpSpec a{"first",{silu[0]}};
  rejects([&]{NonlinearOpRegistry invalid({a,a});},"duplicate activation accepted");
  rejects([&]{NonlinearOpRegistry invalid({a,{"second",{silu[0]}}});},"duplicate global ID accepted");
  auto rejectRecipe=[&](PolyTemplateSpec s) {
    rejects([&]{NonlinearOpRegistry invalid({{"first",{s}}});},"invalid stage recipe accepted");
  };
  auto bad=silu[0]; bad.id.clear(); rejectRecipe(bad);
  bad=silu[0]; bad.stages.clear(); rejectRecipe(bad);
  bad=silu[0]; bad.templateRefError=std::numeric_limits<double>::quiet_NaN(); rejectRecipe(bad);
  bad=silu[0]; bad.templateRefError=-1; rejectRecipe(bad);
  bad=silu[0]; bad.stages={static_cast<F>(123)}; rejectRecipe(bad);

  // Exercise the actual candidate consumer for each migrated template. This
  // is not evidence of a new source witness, new certifier or ReLU lowering.
  for (auto spec:{&silu[0],&silu[1],&gelu[0]}) {
    bool composed=spec->stages.size()>1; DPOptions options;
    options.direct=!composed; options.composed=composed; options.degrees={3};
    auto result=synthesizePolynomials(spec==&gelu[0]?"gelu":"silu",{-0.5,0.5},options);
    expect(!result.frontier.empty(),"registered recipe disappeared from DP");
    for (const auto& c:result.frontier) {
      expect(c.templateName==spec->id && c.stages.size()==spec->stages.size() && std::isfinite(c.approximationError),
             "compiled candidate does not match its registered template");
      for (size_t i=0;i<c.stages.size();++i)
        expect(c.stages[i].stage==stageName(spec->stages[i]),"candidate stage reordered");
    }
  }
}
}  // namespace

int main() {
  try { run(); } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
  std::cout<<"NonlinearOpRegistryTest PASS: three legacy templates; ReLU not admitted\n";
}
