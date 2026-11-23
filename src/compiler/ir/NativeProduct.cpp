#include "NativeProduct.h"
#include "Target/Lattigo/NativeSingleGo.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <filesystem>
#include <set>

namespace snabf::native {
namespace {
using mlir::ModuleOp;
using llvm::StringRef;
constexpr StringRef source = "snabf.n1_source", ports = "snabf.n1_ports",
                    stage = "snabf.n1_stage", plan = "snabf.n1_plan", seal = "snabf.n1_seal",
                    n2 = "snabf.n1_n2_profile";
void need(bool ok, StringRef why) { if (!ok) throw std::invalid_argument(why.str()); }
StringRef field(ModuleOp m, StringRef name) {
  auto a = m->getAttrOfType<mlir::StringAttr>(name);
  need(a && !a.getValue().empty(), "missing/invalid native product field");
  return a.getValue();
}
void set(ModuleOp m, StringRef k, StringRef v) {
  m->setAttr(k, mlir::StringAttr::get(m.getContext(), v));
}
void carrier(ModuleOp m, StringRef state) {
  std::set<StringRef> allowed{source, ports, stage};
  if(m->hasAttr("snabf.n1_profile")) {
    need(!m->hasAttr(n2),"conflicting native ReLU and N2 profiles");
    need(field(m,"snabf.n1_profile")=="relu-affine-2p-v1","unknown native profile");
    allowed.insert("snabf.n1_profile");
  }
  if (m->hasAttr(n2)) allowed.insert(n2);
  if (state != "source") allowed.insert(plan);
  if (state == "checked") allowed.insert(seal);
  need(m.getBody()->empty() && field(m, stage) == state, "wrong native product body/stage");
  for (auto a : m->getAttrs()) need(allowed.erase(a.getName().getValue()) == 1, "unexpected native product attribute");
  need(allowed.empty(), "incomplete native product carrier");
}
void writeNew(const std::filesystem::path &path, StringRef bytes) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path.string(), ec, llvm::sys::fs::CD_CreateNew);
  need(!ec, "cannot create new native output file");
  out << bytes; out.close(); need(!out.has_error(), "native output write failed");
}
struct Rebuilt {
  pass3::Options options;
  NativeArithmeticProgram program;
  pass3::Solution solution;
  size_t attempts = 0;
  const ExactRational budget = ExactRational(1)/(mpz_class(1)<<40); // decoded bias, not probability
  explicit Rebuilt(ModuleOp m) {
    ArithmeticOptions opt;
    opt.domain = {14, 4096}; opt.polynomial.degrees = {3, 5, 7}; opt.polynomial.composed = false;
    options.maxK = 0; opt.activationTolerance = options.tolerance;
    if(m->hasAttr("snabf.n1_profile")) {
      configureNativeRelu(opt,options);
    } else if (m->hasAttr(n2)) {
      auto raw = field(m, n2);
      auto parsed = llvm::json::parse(raw);
      if (!parsed) throw std::invalid_argument(llvm::toString(parsed.takeError()));
      auto *cfg = parsed->getAsObject();
      need(cfg && cfg->size() == 7 && cfg->getString("schema") == "snabf.n2-profile.v1" &&
           llvm::formatv("{0}", *parsed).str() == raw, "noncanonical native N2 profile");
      auto integer = [&](StringRef name, int lo, int hi) -> int {
        auto x = cfg->getInteger(name);
        need(x && *x >= lo && *x <= hi, "invalid native N2 profile coordinate");
        return *x;
      };
      int logN = integer("log_n", 14, 16), n = 1 << logN;
      int slots = integer("slots", 1, n/2);
      need(!(slots & (slots-1)), "invalid native N2 slot count");
      opt.domain = {unsigned(logN), unsigned(slots)};
      opt.qCount = integer("q_count", 2, 32);
      options.alpha = integer("alpha", 1, 2);
      auto mode = cfg->getString("scale_mode");
      need(mode == "narrow" || mode == "wide", "invalid native N2 scale mode");
      options.heterogeneous = mode == "wide";
      int cap = integer("cap_bits", n/128, n/32);
      need(cap % (n/512) == 0, "off-grid native N2 envelope");
      options.securityCap = cap;
      options.minScale = 42; options.maxScale = 49;
    }
    std::string errors;
    llvm::raw_string_ostream diag(errors);
    if (mlir::failed(specializeNativeArithmetic(field(m, source), "original.mlir", field(m, ports),
                    opt, options, 32, program, solution, attempts, diag)))
      throw std::invalid_argument("native source specialization failed: " + errors);
  }
  std::string security(StringRef file) const {
    need(!file.empty() && file != "-", "native-security file required");
    auto buffer = llvm::MemoryBuffer::getFile(file);
    need(bool(buffer), "cannot read native-security");
    auto text = (*buffer)->getBuffer();
    auto parsed = llvm::json::parse(text);
    if (!parsed) throw std::invalid_argument(llvm::toString(parsed.takeError()));
    auto *r = parsed->getAsObject();
    need(r && r->getString("input_spec") == program.source.inspection.getString("input_spec_sha256"),
         "native-security input domain/port mismatch");
    return text.str();
  }
};
template<class F> mlir::LogicalResult diagnosed(ModuleOp m, F action) {
  try { action(); return mlir::success(); }
  catch (const std::exception &e) { return m.emitError(e.what()); }
}
std::string planSeal(StringRef text) {
  // Only called after the whole plan has passed independent semantic replay.
  auto p = llvm::json::parse(text);
  return p->getAsObject()->getString("emit_sha256")->str();
}
} // namespace

bool isProduct(ModuleOp m) {
  for (auto attr : m->getAttrs()) if (attr.getName().getValue().starts_with("snabf.n1_")) return true;
  return false;
}
mlir::LogicalResult planProduct(ModuleOp m, StringRef proposal) {
  return diagnosed(m, [&]{
    carrier(m, "source"); Rebuilt r(m);
    auto p = lowerNativePhysical(r.program, r.solution, r.options, r.budget);
    set(m, plan, llvm::formatv("{0}", llvm::json::Value(llvm::json::Object(p))).str());
    set(m, stage, "planned");
    if (!proposal.empty()) writeNew(proposal.str(), llvm::formatv("{0}\n", llvm::json::Value(llvm::json::Object{
      {"plan", std::move(p)}, {"attempts", int64_t(r.attempts)}, {"security_cap", r.options.securityCap}})).str());
  });
}
mlir::LogicalResult checkProduct(ModuleOp m, StringRef security) {
  return diagnosed(m, [&]{
    bool recheck = field(m, stage) == "checked";
    carrier(m, recheck ? "checked" : "planned"); Rebuilt r(m);
    auto text = field(m, plan);
    checkNativeSingle(r.program, r.solution, r.options, r.budget, text, r.security(security));
    auto actual = planSeal(text);
    if (recheck) need(field(m, seal) == actual, "stale native checked seal");
    set(m, seal, actual); set(m, stage, "checked");
  });
}
mlir::LogicalResult lowerProduct(ModuleOp m, StringRef security, StringRef output) {
  return diagnosed(m, [&]{
    carrier(m, "checked"); Rebuilt r(m);
    auto text = field(m, plan);
    auto model = emitNativeSingleGo(r.program, r.solution, r.options, r.budget, text, r.security(security));
    need(field(m, seal) == planSeal(text), "stale native checked seal");
    if (!output.empty()) {
      std::filesystem::path dir(output.str());
      need(std::filesystem::create_directory(dir), "native-output must be a new directory");
      writeNew(dir/"program.json", text); writeNew(dir/"model.go", model);
      writeNew(dir/"native_runtime.go", nativeSingleRuntimeSource());
    }
    set(m, "snabf.n1_go", model); set(m, stage, "lowered");
  });
}
} // namespace snabf::native
