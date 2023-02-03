#include "tiny/TinyDialect.h"
#include "tiny/TinyOps.h"


#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"
#include "tiny/TinyOpsDialect.cpp.inc"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"

//这里只是引入了interface可以参考的用法，目前这段代码的逻辑还没有被激活
struct ToyInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  // if the given operation is legal to inline into the given region
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // if the given 'src' region can be inlined into the 'dest' region
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final {
    return true;
  }
  void handleTerminator(mlir::Operation *op,
                        llvm::ArrayRef<mlir::Value> valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = mlir::cast<mlir::tiny::ReturnOp>(op);
    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};

// Dialect initialization, the instance will be owned by the context. This is  the point of registration of types adn operations for the dialect.
// 每个Dialect中，至少应该包含一个初始化函数，这个的写法也是固定的
void mlir::tiny::TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny/TinyOps.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}


