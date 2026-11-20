include "mlir/Pass/PassBase.td"

class ModuleStage<string flag, string desc> : Pass<flag,"::mlir::ModuleOp"> {
  let summary = desc;
  let dependentDialects = ["::snabf::ir::SNABF", "::mlir::func::FuncDialect"];
}
def ImportHEIR : ModuleStage<"snabf-import-heir", "Lift the HEIR CKKS evaluator into budget-flow SSA"> {
  let options = [Option<"bytes","key-bytes","int64_t","0","Serialized size of a full Galois key for the input profile">,
    Option<"bound","input-bound","double","1.0","Symmetric input domain">,
    Option<"limit","rotation-path-limit","int64_t","-1","Optional per-sink bound on sequential Pass 1 automorphisms">,
    Option<"hoist","hoist","double","0.625","Hoisted rotation units per active limb">,
    Option<"giant","giant","double","1.0","Giant rotation units per active limb">,
    Option<"eta","key-units-per-mib","double","1.0","Cost per MiB of physical key storage">,
    Option<"pricing","route-pricing","std::string","\"occupied\"","occupied sparse work or literal dense_tower equation (2)">];
}
def Pass1Candidates : ModuleStage<"snabf-pass1-candidates","Publish subgroup-tower routes and evaluate ring-switched bootstrap sites"> {
  let dependentDialects = ["::snabf::ir::SNABF", "::mlir::heir::lattigo::LattigoDialect"];
}
def MarkActivations : ModuleStage<"snabf-mark-activations", "Identify source SiLU semantics before HEIR polynomial approximation"> {
  let options = [Option<"staticEntry", "static-entry", "bool", "false", "Validate the single-input static source subset before HEIR lowering">];
  let dependentDialects = ["::mlir::linalg::LinalgDialect", "::mlir::arith::ArithDialect", "::mlir::math::MathDialect"];
}
def Pass2Candidates : ModuleStage<"snabf-pass2-candidates", "Certify full activation DAGs and publish the Pass 2 Pareto frontier"> {
  let options = [Option<"depth","depth","int64_t","12","Maximum RNS levels for the complete activation">,
    Option<"epsilon","epsilon","double","0.0002","Full activation approximation error in decoded units">,
    Option<"minLevel","min-output-level","int64_t","0","Required suffix output level">,
    Option<"maxError","max-output-error","double","-1","Optional propagated deterministic approximation component at sinks">,
    Option<"family","template-family","std::string","\"all\"","Allowed template family: all, direct, or composed (explicit ablation)">];
}
def Plan : ModuleStage<"snabf-plan","Select candidates over SSA, re-solve violated sink budgets"> {
  let dependentDialects = ["::snabf::ir::SNABF", "::mlir::func::FuncDialect", "::mlir::heir::lattigo::LattigoDialect"];
  let options = [Option<"independent","independent","bool","false","Decide per layer, score physical key union">];
}
def CheckPlan : ModuleStage<"snabf-check-plan","Recompute budget transfers, selected routes and key inventory"> {
  let options = [Option<"requireDown","require-ring-down","bool","false","Reject a checked fallback when a physical down-conversion was requested">];
  let dependentDialects = ["::snabf::ir::SNABF", "::mlir::func::FuncDialect", "::mlir::heir::lattigo::LattigoDialect"];
}
def Pass3Parameters : ModuleStage<"snabf-pass3", "Optimize and concretize RNS, scales and centered-prefix gadget sites"> {
  let options=[Option<"minScale","min-scale","int","36","Smallest default encoding exponent">, Option<"maxScale","max-scale","int","46","Largest default encoding exponent">,
    Option<"maxK","max-k","int","1","Maximum supported centered CRT prefix">, Option<"tolerance","epsilon","double","0.000244140625","Decoded output tolerance">,
    Option<"sourceFraction","source-arithmetic-fraction","double","1.0","Fraction of post-approximation headroom allocated to source parameter synthesis">,
    Option<"cap","security-cap","double","350","Pre-screen QP bits (concrete estimator also required)">,Option<"minBase","min-base","double","24","Minimum q0 bit width">,
    Option<"guard","guard","double","1","Message headroom bits">,Option<"fixed","fixed-exact","bool","false","Exact matched-parameter ablation">,
    Option<"audit","audit","std::string","\"\"","Proposal and concrete arithmetic JSON">,Option<"security","security","std::string","\"\"","Concrete estimator report for this profile and key inventory">];
}
def DynamicRings : ModuleStage<"snabf-pass4", "Replan and independently screen physical ring epochs"> {
  let dependentDialects = ["::snabf::ir::SNABF", "::mlir::func::FuncDialect", "::mlir::heir::lattigo::LattigoDialect"];
  let options = [Option<"minRing","min-ring","int64_t","4096","Minimum proposed physical ring">,
    Option<"maxIters","max-iters","unsigned","16","Maximum unique proposals">,
    Option<"price","switch-price","double","1.0","Relative cost of a cross-key external product">,
    Option<"cutProfile","cut-profile","std::string","\"single-live-v1\"","single-live-v1 or dual-live-v1 (one periodic boundary)">,
    Option<"forceDown","force-down","bool","false","Choose the least-cost feasible down proposal even without net benefit">,
    Option<"audit","audit","std::string","\"\"","Optional decision trace JSON">];
}
def LowerCKKS : ModuleStage<"snabf-lower-ckks","Materialize the selected implementations into HEIR CKKS"> {
  let options = [Option<"directGo","direct-go","bool","false","Preserve checked SSA for the direct Go target (automatic with Pass 4)">];
  let dependentDialects = ["::snabf::ir::SNABF", "::mlir::func::FuncDialect", "::mlir::heir::lattigo::LattigoDialect"];
}
def BindLattigo : ModuleStage<"snabf-bind-lattigo","Bind verified routes and key inventory after HEIR bufferization"> {
  let dependentDialects = ["::mlir::heir::lattigo::LattigoDialect", "::mlir::func::FuncDialect", "::mlir::memref::MemRefDialect"];
  let options = [Option<"audit","audit","std::string","\"\"","Optional JSON export for Go binding and experiment audit">,
    Option<"ringGo","ring-go","std::string","\"\"","Atomically emit the verified straight-line Go target">];
}
