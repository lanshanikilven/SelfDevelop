add_mlir_dialect_library(Graph
  GraphDialect.cpp
  GraphOps.cpp
  LowerGraphPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Graph

  LINK_LIBS PUBLIC
  MLIRIR
  )
  

#add_mlir_library(LowerGraphPass
  #LowerGraphPass.cpp
#)
