#include "DirectEmitter.h"
#include "EmitGraph.h"
#include "Pass2IR.h"
#include "Pass3IR.h"
#include "Pass4IR.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <filesystem>

namespace snabf::ir::lattigo_target {
namespace {
constexpr StringLiteral seal = "snabf.emit_sha256", epochs = "snabf.ckks_epochs",
                        single = "snabf.ckks_direct";
StringRef artifact(ModuleOp m) {
  return isMixed(m) || m->hasAttr("snabf.ring_search") ? epochs : single;
}
Attribute existing(ModuleOp m) {
  StringRef expected=artifact(m);
  require(!m->hasAttr(expected==epochs?single:epochs),"wrong_lowering_domain");
  return m->getAttr(expected);
}

void replay(ModuleOp m) {
  if (isMixed(m)) checkRings(m);
  else { verifyPlan(m); verifyPass3(m); checkRingSearch(m); }
}

std::string fingerprint(ModuleOp m) {
  // Bind the entire replay input, including signed weights, SSA uses,
  // concrete security evidence and any new Pass 4 domain/cut records. Ignore
  // locations, the seal itself and the derived lowering so it survives MLIR I/O.
  OwningOpRef<ModuleOp> copy(cast<ModuleOp>(m->clone()));
  (*copy)->removeAttr(seal);
  (*copy)->removeAttr(epochs);
  (*copy)->removeAttr(single);
  std::string text = "snabf.direct-go.v1\n";
  llvm::raw_string_ostream os(text);
  (*copy)->print(os, OpPrintingFlags().printGenericOpForm().useLocalScope());
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}

DictionaryAttr request(ModuleOp m) {
  require(m->hasAttr("snabf.pass3") && m->hasAttr("snabf.security_screen"),
          "direct_target_requires_concrete_screened_parameters");
  auto proof = field<StringAttr>(m, seal);
  replay(m);
  require(proof.getValue() == fingerprint(m), "stale_emit_fingerprint");
  // Unsupported operations may not disappear between the plan and Go.
  m.walk([&](Operation *o) {
    if (auto s = o->getAttrOfType<DictionaryAttr>("snabf.ring_switch"))
      require(!field<BoolAttr>(s, "enabled").getValue(),
              "bootstrap_not_emittable_in_epoch_graph");
  });
  return detail::bindEncodingProgram(m, ringProgram(m), false);
}

// Same-directory temporaries ensure rename is atomic. The caller stages both
// outputs and publishes the Go source last; validation never touches outputs.
class PendingFile {
  llvm::sys::fs::TempFile tmp;
  std::string target;
  bool finished = false;
  static llvm::sys::fs::TempFile create(StringRef path) {
    auto f = llvm::sys::fs::TempFile::create(path + ".%%%%%%.tmp");
    if (!f) throw ContractError("cannot_stage_output: " + llvm::toString(f.takeError()));
    return std::move(*f);
  }
public:
  PendingFile(StringRef path, StringRef contents) : tmp(create(path)), target(path.str()) {
    llvm::raw_fd_ostream out(tmp.FD, false);
    out << contents;
    out.flush();
    if (out.has_error()) {
      auto msg = out.error().message(); out.clear_error();
      llvm::consumeError(tmp.discard());
      throw ContractError("cannot_stage_output: " + msg);
    }
  }
  void publish() {
    if (auto err = tmp.keep(target)) throw ContractError("cannot_publish_output: " + llvm::toString(std::move(err)));
    finished = true;
  }
  ~PendingFile() { if (!finished) llvm::consumeError(tmp.discard()); }
};
} // namespace

void checkAndSeal(ModuleOp m) {
  replay(m);
  if (!m->hasAttr("snabf.pass3")) return; // legacy HEIR path
  if (!m->hasAttr(seal))
    require(!existing(m) && !m->hasAttr("snabf.encoding_receipts"), "unsealed_derived_encoding_program");
  auto program = detail::bindEncodingProgram(m, ringProgram(m), !m->hasAttr(seal));
  auto hash = fingerprint(m);
  if (m->hasAttr(seal))
    require(field<StringAttr>(m, seal).getValue() == hash, "stale_emit_fingerprint");
  if (auto p = existing(m))
    require(p == program, "stale_epoch_lowering");
  m->setAttr(seal, StringAttr::get(m.getContext(), hash));
}
void lowerDirect(ModuleOp m) {
  auto p = request(m);
  if (auto old = existing(m)) require(old == p, "stale_epoch_lowering");
  m->setAttr(artifact(m), p);
}
void emitDirect(ModuleOp m, StringRef goPath, StringRef auditPath) {
  require(!goPath.empty() && goPath != "-", "direct_target_requires_ring_go_output_file");
  namespace fs=std::filesystem;
  auto regularOrAbsent=[](StringRef name){
    auto st=fs::symlink_status(name.str());
    require(st.type()==fs::file_type::not_found || fs::is_regular_file(st), "unsafe_emitter_output: "+name);
  };
  regularOrAbsent(goPath);
  if (!auditPath.empty()) {
    require(auditPath!="-","direct_target_requires_audit_file"); regularOrAbsent(auditPath);
    require(fs::weakly_canonical(fs::absolute(goPath.str()))!=fs::weakly_canonical(fs::absolute(auditPath.str())) &&
            !(fs::exists(goPath.str()) && fs::exists(auditPath.str()) && fs::equivalent(goPath.str(),auditPath.str())),
            "aliased_go_and_audit_output");
  }
  auto p = request(m);
  require(existing(m) == p, "stale_epoch_lowering");
  auto hash = field<StringAttr>(m, seal).getValue();
  auto go = detail::renderGo(p, hash);
  PendingFile code(goPath, go);
  if (!auditPath.empty()) {
    auto audit = jsonPass2(p);
    audit["emit_sha256"] = hash.str();
    audit["emitter"] = "snabf.direct-go.v1";
    auto text = llvm::formatv("{0:2}\n", llvm::json::Value(std::move(audit))).str();
    PendingFile record(auditPath, text);
    // The Go file is the publication point. Restore the previous sidecar if
    // its publication succeeded but the final rename failed. Readers spanning
    // both renames must compare the SHA printed in both files.
    std::unique_ptr<PendingFile> oldAudit;
    if (fs::exists(auditPath.str())) {
      auto buf=llvm::MemoryBuffer::getFile(auditPath);
      require(bool(buf),"cannot_preserve_previous_audit");
      oldAudit=std::make_unique<PendingFile>(auditPath,(*buf)->getBuffer());
    }
    record.publish();
    try { code.publish(); }
    catch (...) {
      if (oldAudit) oldAudit->publish();
      else if(auto ec=llvm::sys::fs::remove(auditPath)) throw ContractError("cannot_restore_audit: "+ec.message());
      throw;
    }
    return;
  }
  code.publish();
}
} // namespace snabf::ir::lattigo_target
