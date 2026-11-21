#include "Pipeline.h"
#include "NativeContracts.h"
#include "Pass2IR.h"
#include "Pass3IR.h"
#include "pass3/CostProfile.h"
#include "NoiseCertIR.h"
#include "Pass4IR.h"
#include "Target/Lattigo/DirectEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"

namespace snabf::ir {
#define GEN_PASS_DECL
#include "Passes.h.inc"
#define GEN_PASS_DEF_IMPORTHEIR
#define GEN_PASS_DEF_PASS1CANDIDATES
#define GEN_PASS_DEF_MARKACTIVATIONS
#define GEN_PASS_DEF_PASS2CANDIDATES
#define GEN_PASS_DEF_PASS3PARAMETERS
#define GEN_PASS_DEF_DYNAMICRINGS
#define GEN_PASS_DEF_PLAN
#define GEN_PASS_DEF_CHECKPLAN
#define GEN_PASS_DEF_LOWERCKKS
#define GEN_PASS_DEF_BINDLATTIGO
#include "Passes.h.inc"

template <class F>
static LogicalResult diagnosed(ModuleOp module, F &&action) {
  try {
    if (failed(rejectNativeExecution(module))) return failure();
    action();
    return success();
  } catch (const std::exception &e) {
    module.emitError(e.what());
    return failure();
  }
}
struct Import : impl::ImportHEIRBase<Import> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(diagnosed(getOperation(), [&] {
          require(limit >= -1, "invalid sink limit");
          importCKKS(getOperation(), bytes, bound, hoist, giant, eta);
          require(pricing=="occupied" || pricing=="dense_tower","invalid route-pricing");
          getOperation()->setAttr("snabf.route_pricing", StringAttr::get(&getContext(),pricing));
          if (getOperation()->hasAttr("snabf.activation_sources"))
            liftActivationRegions(getOperation());
          if (limit >= 0)
            getOperation().walk([&](SinkOp op) {
              op->setAttr(
                  "rotation_path_limit",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64), limit));
            });
        })))
      signalPassFailure();
  }
};
struct Marker : impl::MarkActivationsBase<Marker> {
  using Base::Base;
  void runOnOperation() override {
    if (staticEntry && failed(verifyStaticSource(getOperation()))) {
      signalPassFailure();
      return;
    }
    if (failed(diagnosed(getOperation(), [&] {
          auto m = getOperation();
          markActivationSources(m);
          if (!receipt.empty()) {
            std::error_code ec;
            llvm::raw_fd_ostream out(receipt,ec,llvm::sys::fs::OF_Text);
            require(!ec,"cannot write activation receipt");
            llvm::json::Object flags{{"schema","snabf.activation-pipeline.v1"},
              {"protect_erf",bool(m->getAttrOfType<UnitAttr>("snabf.protect_erf"))}};
            out << llvm::formatv("{0}\n",llvm::json::Value(std::move(flags)));
            out.flush(); require(!out.has_error(),"activation receipt write failed");
          }
          if (staticEntry && !m->hasAttr("snabf.initial_profile"))
            m->setAttr("snabf.initial_profile", StringAttr::get(&getContext(), "epoch_0"));
        }))) signalPassFailure();
  }
};
struct Pass2 : impl::Pass2CandidatesBase<Pass2> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(diagnosed(getOperation(), [&] {
      if (!intervalMode.empty()) {
        require(intervalMode == "naive" || intervalMode == "affine", "invalid snabf.interval_mode");
        getOperation()->setAttr("snabf.interval_mode", StringAttr::get(&getContext(), intervalMode));
      }
      publishPolynomials(getOperation(),depth,epsilon,minLevel,maxError,family);
    }))) signalPassFailure();
  }
};
struct Pass1 : impl::Pass1CandidatesBase<Pass1> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(
            diagnosed(getOperation(), [&] { publishPass1(getOperation()); })))
      signalPassFailure();
  }
};
struct Parameters3 : impl::Pass3ParametersBase<Parameters3> {
 using Base::Base;
 void runOnOperation() override {
   if (failed(diagnosed(getOperation(),[&]{
     pass3::Options options{minScale,maxScale,maxK,tolerance,cap,minBase,guard,gadgetAlpha,hetero,heirAnchor,widthMin,widthMax,auxMargin};
     require(costProfileFile.empty()==costHardware.empty(), "cost_profile_and_hardware_id_required_together");
     if (!costProfileFile.empty()) {
       require(options.blockSearch(), "calibrated_cost_requires_explicit_gadget_alpha");
       options.costProfile=pass3::CostProfile::loadFile(costProfileFile.getValue(),costHardware.getValue());
     }
     publishPass3(getOperation(),options,fixed,audit,security,sourceFraction);
   }))) signalPassFailure();
 }
};
struct Rings : impl::DynamicRingsBase<Rings> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(diagnosed(getOperation(), [&]{planRings(getOperation(),minRing,maxIters,price,audit,cutProfile,forceDown);}))) signalPassFailure();
  }
};
struct Planner : impl::PlanBase<Planner> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(diagnosed(getOperation(),
                         [&] { planModule(getOperation(), independent); })))
      signalPassFailure();
  }
};
struct Checker : impl::CheckPlanBase<Checker> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(diagnosed(getOperation(), [&] {
          auto m = getOperation();
          lattigo_target::checkAndSeal(m);
          if (requireDown) requireCheckedRingDown(m);
          if (!noiseAudit.empty() || requireNoiseCert)
            auditNoisePlan(m,noiseAudit,requireNoiseCert,precisionBits);
        })))
      signalPassFailure();
  }
};
struct Lower : impl::LowerCKKSBase<Lower> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(diagnosed(getOperation(), [&] {
          if (directGo) lattigo_target::lowerDirect(getOperation());
          else lowerCKKS(getOperation());
        })))
      signalPassFailure();
  }
};
struct Bind : impl::BindLattigoBase<Bind> {
  using Base::Base;
  void runOnOperation() override {
    if (failed(diagnosed(getOperation(), [&] {
          auto m=getOperation();
          if(!ringGo.empty() || isMixed(m) || m->hasAttr("snabf.ring_search") || m->hasAttr("snabf.ckks_epochs") || m->hasAttr("snabf.ckks_direct")) {
            lattigo_target::emitDirect(m, ringGo, audit); return;
          }
          auto result=bindBackend(m);
          if (!audit.empty()) {
            std::error_code ec;
            llvm::raw_fd_ostream os(audit, ec, llvm::sys::fs::OF_Text);
            require(!ec, "cannot write backend audit");
            os << llvm::formatv("{0:2}\n",
                                llvm::json::Value(std::move(result)));
          }
        })))
      signalPassFailure();
  }
};
void registerPasses() {
  PassRegistration<Import>();
  PassRegistration<Pass1>();
  PassRegistration<Marker>();
  PassRegistration<Pass2>();
  PassRegistration<Parameters3>();
  PassRegistration<Rings>();
  PassRegistration<Planner>();
  PassRegistration<Checker>();
  PassRegistration<Lower>();
  PassRegistration<Bind>();
}
}  // namespace snabf::ir
