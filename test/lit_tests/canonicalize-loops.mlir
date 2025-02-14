// RUN: enzymexlamlir-opt --canonicalize-loops -split-input-file %s | FileCheck %s

func.func private @use_value(%arg0: i32)

// CHECK-LABEL: foo
func.func private @foo(%arg0: !llvm.ptr<1>, %true: i1) {
    // CHECK: affine.parallel (%{{.+}}, %{{.+}}) = (0, 0) to (23, 256)
    affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (23, 1, 1, 256, 1, 1) {
      %0 = arith.constant 1 : i32
      scf.if %true {
        %1 = arith.addi %0, %0 : i32
        func.call @use_value(%1) : (i32) -> ()
      }
    }
    return
}

// -----

func.func private @use_value(%arg0: i32)

// CHECK-LABEL: use_indices
func.func private @use_indices() {
    // CHECK: affine.parallel (%[[ARG1:.*]], %[[ARG4:.*]]) = (0, 0) to (23, 256)
    affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (23, 1, 1, 256, 1, 1) {
      // CHECK: %[[IDX1:.*]] = arith.index_cast %[[ARG1]]
      // CHECK: %[[IDX2:.*]] = arith.index_cast %[[ARG4]]
      // CHECK: %[[SUM:.*]] = arith.addi %[[IDX1]], %[[IDX2]]
      // CHECK: func.call @use_value(%[[SUM]])
      %0 = arith.index_cast %arg1 : index to i32
      %1 = arith.index_cast %arg4 : index to i32
      %2 = arith.addi %0, %1 : i32
      func.call @use_value(%2) : (i32) -> ()
    }
    return
}
