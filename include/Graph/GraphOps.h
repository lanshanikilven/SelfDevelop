#ifndef Graph_GraphOPS_H
#define Graph_GraphOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Graph/GraphOps.h.inc"

#endif // Graph_GraphOPS_H