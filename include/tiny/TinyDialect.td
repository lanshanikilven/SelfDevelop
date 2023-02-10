#ifndef TINY_DIALECT
#define TINY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def Tiny_Dialect : Dialect {
    let name = "tiny";
    let summary = "a tiny dialect";
    let cppNamespace = "::mlir::tiny";
    //let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//
//Trait及其子类，如OpTrait等，用于指定op的特殊属性和约束
class Tiny_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Tiny_Dialect, mnemonic, traits>;


#endif // STANDALONE_DIALECT
