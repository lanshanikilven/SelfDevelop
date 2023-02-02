#include "tiny/TinyDialect.h"
#include "tiny/TinyOps.h"


#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "tiny/TinyOpsDialect.cpp.inc"


// Dialect initialization, the instance will be owned by the context. This is  the point of registration of types adn operations for the dialect.
void mlir::tiny::TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny/TinyOps.cpp.inc"
      >();
}


