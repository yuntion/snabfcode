def CT : TypeDef<SNABF, "Ct"> {
  let mnemonic = "ct";
  let parameters = (ins "::mlir::RankedTensorType":$logical, "int64_t":$n,
                        "int64_t":$level, "::mlir::FlatSymbolRefAttr":$profile);
  let assemblyFormat = "`<` $logical `,` $n `,` $level `,` $profile `>`";
  let genVerifyDecl = 1;
}
def Budget : AttrDef<SNABF, "Budget"> {
  let mnemonic = "budget";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}
def SlotLayout : AttrDef<SNABF, "SlotLayout"> {
  let mnemonic = "slot_layout";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}
