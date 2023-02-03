#ifndef TINY_TINYOPS_H
#define TINY_TINYOPS_H

include "mlir/IR/OpBase.td"
include "tiny/TinyDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

/*
def CastOp : Tiny_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    NoMemoryEffect,
    SameOperandsAndResultShape]
  > {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked,
    then shape is required to match. The operation is invalid if converting
    to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
*/

def TransposeOp : Tiny_Op<"transpose", [NoSideEffect]> {
  let summary = "transpose operation";

  let arguments = (ins I32Tensor:$input);
  let results = (outs I32Tensor);
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<(ins  "mlir::Value": $input)>
  ];
  let hasCanonicalizer = 1;
  // Invoke a static verify method to verify this transpose operation.
  //let verifier = [{ return ::verify(*this); }];
}


// Pirnt Operation defnination
def PrintOp : Tiny_Op<"print">  {
  let summary =  "print operation";
  let description = [{
      The print builtin operation prints a given input string, and produces
      nothing.
  }];

  let arguments = (ins AnyTypeOf<[I32Tensor, I32MemRef]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";

  let printer = [{ return ::print(printer,  *this); }];
  let parser  = [{ return ::parser$cppClass(paser, result); }];
}


// We define a toy operation by inheriting from our base 'Toy_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'NoSideEffect' as it is a pure operation
// and may be removed if dead.
def ConstantOp : Tiny_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I32ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs I32Tensor);

  // Specify a parser and printer method.
  let parser = [{ return ::parseConstantOp(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
  //let hasCanonicalizer = 1;
  //let hasFolder = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "int":$value)>
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

def ReturnOp : Tiny_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        toy.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
}

#endif //TINY_TINYOPS_H