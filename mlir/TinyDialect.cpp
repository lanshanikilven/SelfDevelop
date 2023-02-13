#include "tiny/TinyDialect.h"
#include "tiny/TinyOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"
#include "tiny/TinyOpsDialect.cpp.inc"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"

#include "tiny/TinyOps.cpp.inc"
//这部分代码为Toy Operation定义了inline的接口和表达式变形规则，两个isLegalToInline重载函数是两个hook
struct ToyInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  // if the given operation is legal to inline into the given region
  // 第一个hook用来检查给定的可调用操作callable内联到给定调用call中是否合法，检查是否可以内联。
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // if the given 'src' region can be inlined into the 'dest' region
  // 第二个hook用来检查给定的操作是否合法地内联到给定的区域。
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final {
    return true;
  }

  // handleTerminator函数只是处理tiny.return，将返回操作的操作数it.index()直接用返回值it.value()代替
  void handleTerminator(mlir::Operation *op,
                        llvm::ArrayRef<mlir::Value> valuesToRepl) const final {
    // Only "tiny.return" needs to be handled here.
    auto returnOp = mlir::cast<mlir::tiny::ReturnOp>(op);
    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
  //这个函数是內联pass的入口
  mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input,
                                       mlir::Type resultType,
                                       mlir::Location conversionLoc) const final {
    return builder.create<mlir::tiny::CastOp>(conversionLoc, resultType, input);
  }

};
 //这一段关于interface的处理也是后面加的，原生的代码里面暂时不会用到这些相对高级的用法

// Dialect initialization, the instance will be owned by the context. This is  the point of registration of types adn operations for the dialect.
// 每个Dialect中，至少应该包含一个初始化函数，这个的写法也是固定的
void mlir::tiny::TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny/TinyOps.cpp.inc"
      >();
  // 这里将上面的ToyInlinerInterface注册到內联pass中
  addInterfaces<ToyInlinerInterface>(); //这里的addInterfaces并不是对于每种op都是必须的
}

