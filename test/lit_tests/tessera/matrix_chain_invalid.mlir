// RUN: enzymexlamlir-opt %s -tessera-optimize-matrix-chain -verify-diagnostics | FileCheck %s

!a = !llvm.array<2 x array<3 x f64>>
!b = !llvm.array<3 x array<4 x f64>>
!c = !llvm.array<2 x array<4 x f64>>
!d = !llvm.array<4 x array<5 x f64>>
!bc = !llvm.array<3 x array<5 x f64>>
!r = !llvm.array<2 x array<5 x f64>>
module {
  // expected-warning @+1 {{ignoring tessera.reassociate_matmul: expected a pure binary product}}
  tessera.define private @impure(!a, !b) -> !c attributes {argModes = [unit, unit], pure = false, tessera.reassociate_matmul}
  // expected-warning @+1 {{ignoring tessera.reassociate_matmul: expected a pure binary product}}
  tessera.define private @bad_dimensions(!a, !a) -> !c attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // expected-warning @+1 {{ignoring tessera.reassociate_matmul: expected a pure binary product}}
  tessera.define private @bad_result(!a, !b) -> !b attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // expected-warning @+1 {{ignoring tessera.reassociate_matmul: expected a pure binary product}}
  tessera.define private @pointers(!llvm.ptr, !llvm.ptr) -> !llvm.ptr attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // expected-warning @+1 {{ignoring tessera.reassociate_matmul: expected a pure binary product}}
  tessera.define private @wrong_attribute(!a, !b) -> !c attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul = false}

  tessera.define private @finish(!c, !d) -> !r attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @bc(!b, !d) -> !bc attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  tessera.define private @a_bc(!a, !bc) -> !r attributes {argModes = [unit, unit], pure = true, tessera.reassociate_matmul}
  // CHECK-LABEL: llvm.func @keep_side_effects
  // CHECK-NEXT: %[[AB:.*]] = tessera.call @impure(%arg0, %arg1)
  // CHECK-NEXT: %[[R:.*]] = tessera.call @finish(%[[AB]], %arg2)
  // CHECK-NEXT: llvm.return %[[R]]
  llvm.func @keep_side_effects(%a: !a, %b: !b, %d: !d) -> !r {
    %ab = tessera.call @impure(%a, %b) : (!a, !b) -> !c
    %r = tessera.call @finish(%ab, %d) : (!c, !d) -> !r
    llvm.return %r : !r
  }
}
