#ifndef TINY_TINYOPS_H
#define TINY_TINYOPS_H

include "mlir/IR/OpBase.td"
include "tiny/TinyDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//include "tiny/ShapeInferenceInterface.td"

/*
//每个Op在定义后，tablegen会根据Op的参数和返回值自动生成一些builder方法。
//在编译之后，会在builder目录下的include文件夹中的**.h.inc文件中自动产生对应的static 的builder方法的声明，并不需要实例化对象再去调用。
//但是如果默认生成的builder方法中没有我们需要使用的方法，则我们需要在td文件中自定义let builders方法
//为Mul Operation添加了形状推导的特征
def MulOp : Tiny_Op<"mul", [NoSideEffect, 
  //DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
  ]> {
  let summary = "element wise mul operation";
  let description = [{
    The "mul" operation performs eltwise mul between two tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I32Tensor:$lhs, I32Tensor:$rhs);
  let results = (outs I32Tensor:$output);
  //let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let parser  = [{ return ::parserBinaryOp(parser, result); }];
  let printer = [{ return ::printBinaryOp(p,  *this); }];
  //下面这个语句生成对应的builder方法声明为：static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, ::mlir::Value lhs, ::mlir::Value rhs)
  //为了能够通过该方法构建Operation，必须向​​state​​​填充该Operation的属性，操作数，域和返回值类型。
  let builders = [
    OpBuilder<(ins  "mlir::Value": $lhs, "mlir::Value": $rhs)>
  ];
}
*/

//在函数调用时，输入张量的类型是确定的。但在函数声明/定义的时候，输入张量的类型是不确定的。
//因此在调用的时候就需要一个隐藏的数据类型转换，否则无法进行内联操作，因此这里引入了一个cast。
//cast操作可以将确定的数据类型转换为函数期望的数据类型。
//DeclareOpInterfaceMethods这个特征说明程序会识别cast操作。
def CastOp : Tiny_Op<"cast", [
  //DeclareOpInterfaceMethods封装的是专用类trait，需要显式调用
    DeclareOpInterfaceMethods<CastOpInterface>,
    //DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    //下面这两个是通用类trait，对所有op都可以使用，不需要专门为op定义相应的接口，而且会被mlir内部直接调用
    NoSideEffect,
    SameOperandsAndResultShape]
    //如下示例所示：
    //Operation *op=...;
    //if (op->hasTrait<NoSideEffect>()) ...
    //if (isa<ShapeInferenceOpInterface>()) ...
  > {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked,
    then shape is required to match. The operation is invalid if converting
    to a mismatching constant dimension.
  }];

  let arguments = (ins I32Tensor:$input);
  let results = (outs I32Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)"; //这个定义了输出的汇编格式
}


def ReshapeOp : Tiny_Op<"reshape", [NoSideEffect]> {
  let summary = "reshape operation";

  let arguments = (ins I32Tensor:$arg);
  let results = (outs I32Tensor:$res);
  let assemblyFormat = [{
    `(` $arg `:` type($arg) `)` attr-dict `to` type(results)
  }];
  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<(ins  "mlir::Value": $arg)>
  ];
  let hasCanonicalizer = 1;
}


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
  let arguments = (ins Variadic<I32Tensor>:$input);

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

// 解决Reshape(Reshape(x)) = Reshape(x)产生的冗余代码。
// 对于多次相同的张量变形操作，只需要执行一次
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

// 当reshape的输入和输出的类型一样时，说明这个reshape操作是冗余的，直接返回输入的参数即可。即reshape(x)=x
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
// 即当0.getType()与1.getType()相同时，该reshape是冗余的，可以使用操作数$arg代替
def RedundantReshapeOptPattern : Pat<(ReshapeOp:$res $arg), (replaceWithValue $arg), [(TypesAreIdentical $res, $arg)]>;
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<(ReshapeOp:$res (ConstantOp $arg)), (ConstantOp (ReshapeConstant $arg, $res))>;


// 需要让內联器知道IR中的tiny.generic_call表示的是调用一个函数
// MLIR提供了一个Operation接口CallOpInterface可以将某个Operation标记为调用。
// 使用了DeclareOpInterfaceMethods在CallOpInterface的声明中 声明采用的接口方法
def GenericCallOp : Tiny_Op<"generic_call", [DeclareOpInterfaceMethods<CallOpInterface>] > {
  let summary =  "generic call operation";
  let description = [{
      The generic call operation can be expand in the positon where it is called.
  }];

  let arguments = (ins FlatSymbolRefAttr : $callee, Variadic<I32Tensor> : $inputs);
  let results = (outs I32Tensor); 
  let assemblyFormat = [{ $callee `(` $inputs `)`  attr-dict  `:` functional-type($inputs,  results) }];
  
  let builders = [
    OpBuilder<(ins "StringRef " : $callee, "ArrayRef<Value>" : $arguments) >
  ];
}



#endif //TINY_TINYOPS_H