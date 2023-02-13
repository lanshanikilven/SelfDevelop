#include "tiny/TinyOps.h"
#include "tiny/TinyDialect.h"
#include "tiny/Passes.h"
#include "tiny/ShapeInferenceInterface.cpp.inc"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/PointerUnion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dialect.h"

#include <iostream>

using namespace mlir;
using namespace mlir::tiny;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "tiny/TinyOps.inc"
}

//ShapeInferencePass继承了FunctionPass，重写其runOnFunction()接口，实现Shape推断算法。
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, FunctionPass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!opWorklist.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      // Ask the operation to infer its output shapes.
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    //if (!opWorklist.empty()) {
    //  f.emitError("Shape inference failed, ")
         // << opWorklist.size() << " operations couldn't be inferred\n";
      //signalPassFailure();
    //}
  }

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
  static bool allOperandsInferred(Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
      return operandType.isa<RankedTensorType>();
    });
  }

  /// A utility method that returns if the given operation has a dynamically
  /// shaped result.
  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return !resultType.isa<RankedTensorType>();
    });
  }
};

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::tiny::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}


// fold transpose(transpose(x)) = x
// 匹配该IR中的所有 toy.transpose
struct SimplifyRedundantTranspose
    : public mlir::OpRewritePattern<tiny::TransposeOp> {
  using OpRewritePattern<tiny::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tiny::TransposeOp op,
                                PatternRewriter &rewriter) const final {
    mlir::Value transposeInput = op.getOperand();

    tiny::TransposeOp transposeInputOp =
        transposeInput.getDefiningOp<tiny::TransposeOp>();
    if (!transposeInputOp) {
      return failure();
    }

    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

void TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                              MLIRContext *context) {
  // SimplifyRedundantTranspose 就是上面定义的结构体(类)
  results.insert<SimplifyRedundantTranspose>(context);
}

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  state.addOperands(value);
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}

//===----------------------------------------------------------------------===//
// ConstantOp

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       int value) {
  auto dataType = RankedTensorType::get({}, builder.getI32Type());
  auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::DenseIntElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
static void print(mlir::OpAsmPrinter &printer, ConstantOp op) {
  printer << "tiny.constant ";
  printer.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << op.value();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(ConstantOp op) {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = op.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType)
    return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op.emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op.emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ReturnOp

static mlir::LogicalResult verify(ReturnOp op) {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(op->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError()
           << "does not return the same number of values ("
           << op.getNumOperands() << ") as the enclosing function ("
           << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!op.hasOperand())
    return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return op.emitError() << "type of return operand (" << inputType
                        << ") doesn't match function result type ("
                        << resultType << ")";
}


//下面两个函数实现了GenericCallOp类的一些功能
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  state.addOperands(arguments);
  state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}


// GenericCallOp::getCallableForCallee() {...} 返回泛化调用Operation的被调用方
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

// GenericCallOp::getArgOperands(){...}用来获取被调用函数的参数操作数。
mlir::Operation::operand_range GenericCallOp::getArgOperands() {
  return inputs();
}

//这个方法用来判断是否需要进行类型转换，如果inputs和outputs的类型是兼容的则返回真，否则需要进行类型转换（cast）返回假。
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = inputs.front().dyn_cast<TensorType>();
  TensorType output = outputs.front().dyn_cast<TensorType>();
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}


void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  state.addOperands({lhs, rhs});
}

//需要进行形状推导的每个Operation，都需要定义对应的inferShapes()函数，比如MulOp，结果的形状就是输入的形状
void mlir::tiny::MulOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//需要进行形状推导的每个Operation，都需要定义对应的inferShapes()函数，比如CastOp，结果的形状就是输入的形状
void mlir::tiny::CastOp::inferShapes() { getResult().setType(getOperand().getType()); }

void mlir::tiny::TransposeOp::inferShapes() {
  auto arrayTy = getOperand().getType().cast<RankedTensorType>();
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

//void ShapeInference::inferShapes() { getResult().setType(getOperand().getType()); }


/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  mlir::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  mlir::Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (mlir::FunctionType funcType = type.dyn_cast<mlir::FunctionType>()) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << op->getName() << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  mlir::Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](mlir::Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }
  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}



// 这里是引入了所有TinyOps
#define GET_OP_CLASSES
#include "tiny/TinyOps.cpp.inc"
