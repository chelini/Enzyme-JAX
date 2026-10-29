// RUN: enzymexlamlir-opt %s -parse-optimization-rules -cse -tessera-propagate-properties -tessera-apply-pdl | FileCheck %s --check-prefix=REWRITE
// RUN: enzymexlamlir-opt %s -parse-optimization-rules -cse -tessera-propagate-properties -tessera-apply-pdl -tessera-lower-guards | FileCheck %s --check-prefix=LOWER

// Linnea trick4, including repeated loads introduced by LLVM-to-Tessera.
!a = !llvm.array<4 x array<3 x f64>>
!at = !llvm.array<3 x array<4 x f64>>
!c = !llvm.array<16 x f64>
!out = !llvm.array<3 x array<3 x f64>>
module {
  tessera.define private @su.transpose(!a) -> !at attributes {argModes = [unit], pure = true}
  tessera.define private @su.mm(!at, !a) -> !out attributes {argModes = [unit, unit], pure = true}
  tessera.define private @su.ca(%c: !c {tessera.layout = {elem = f64, rows = 4 : i64, cols = 4 : i64, row_major = true}}, %a: !a) -> !a attributes {argModes = [unit, unit], pure = true}
  tessera.define private @su.add(!out, !out) -> !out attributes {argModes = [unit, unit], pure = true}
  tessera.define private @su.axpy(f64, !a, !a) -> !a attributes {argModes = [unit, unit, unit], pure = true}
  tessera.define private @su.syr2k(!a, !a) -> !out attributes {argModes = [unit, unit], pure = true}
  tessera.optimizations {
    tessera.optimization "if symmetric(c), su.add(su.add(su.mm(su.transpose(a), b), su.mm(su.transpose(b), a)), su.mm(su.transpose(a), su.ca(c, a))) -> su.syr2k(a, su.axpy(0.5, su.ca(c, a), b))"
  }

  // REWRITE-LABEL: llvm.func @known
  // REWRITE: %[[HALF:.*]] = llvm.mlir.constant(5.000000e-01 : f64)
  // REWRITE: %[[A:.*]] = llvm.load %arg0
  // REWRITE-NOT: llvm.load
  // REWRITE-NOT: tessera.call @su.transpose
  // REWRITE-NOT: tessera.call @su.mm
  // REWRITE: %[[CA:.*]] = tessera.call @su.ca(%arg2, %[[A]])
  // REWRITE-NEXT: %[[T:.*]] = tessera.call @su.axpy(%[[HALF]], %[[CA]], %arg1)
  // REWRITE-NEXT: %[[R:.*]] = tessera.call @su.syr2k(%[[A]], %[[T]])
  // REWRITE-NEXT: llvm.return %[[R]]
  // LOWER-LABEL: llvm.func @known
  // LOWER-NOT: llvm.cond_br
  // LOWER: tessera.call @su.syr2k
  // LOWER-NEXT: llvm.return
  llvm.func @known(%ap: !llvm.ptr, %b: !a, %c: !c {tessera.assumes = ["symmetric"]}) -> !out {
    %a0 = llvm.load %ap : !llvm.ptr -> !a
    %at = tessera.call @su.transpose(%a0) : (!a) -> !at
    %bt = tessera.call @su.transpose(%b) : (!a) -> !at
    %atb = tessera.call @su.mm(%at, %b) : (!at, !a) -> !out
    %a1 = llvm.load %ap : !llvm.ptr -> !a
    %bta = tessera.call @su.mm(%bt, %a1) : (!at, !a) -> !out
    %sum = tessera.call @su.add(%atb, %bta) : (!out, !out) -> !out
    %a2 = llvm.load %ap : !llvm.ptr -> !a
    %ca = tessera.call @su.ca(%c, %a2) : (!c, !a) -> !a
    %atca = tessera.call @su.mm(%at, %ca) : (!at, !a) -> !out
    %result = tessera.call @su.add(%sum, %atca) : (!out, !out) -> !out
    llvm.return %result : !out
  }

  // The optimized and original computations both remain behind the guard.
  // REWRITE-LABEL: llvm.func @checked
  // REWRITE: tessera.guard "symmetric(c)"
  // REWRITE: tessera.call @su.ca
  // REWRITE-NEXT: {{.*}}tessera.call @su.axpy
  // REWRITE-NEXT: {{.*}}tessera.call @su.syr2k
  // REWRITE: } else {
  // REWRITE: tessera.call @su.transpose
  // REWRITE: tessera.call @su.mm
  // REWRITE: tessera.call @su.add
  // REWRITE: llvm.return
  // LOWER-LABEL: llvm.func @checked
  // LOWER-COUNT-6: llvm.fcmp "oeq"
  // LOWER: llvm.cond_br
  // LOWER: tessera.call @su.syr2k
  // LOWER: llvm.br
  // LOWER: tessera.call @su.transpose
  // LOWER: tessera.call @su.mm
  // LOWER: tessera.call @su.add
  // LOWER: llvm.return
  llvm.func @checked(%a: !a, %b: !a, %c: !c) -> !out {
    %at = tessera.call @su.transpose(%a) : (!a) -> !at
    %bt = tessera.call @su.transpose(%b) : (!a) -> !at
    %atb = tessera.call @su.mm(%at, %b) : (!at, !a) -> !out
    %bta = tessera.call @su.mm(%bt, %a) : (!at, !a) -> !out
    %sum = tessera.call @su.add(%atb, %bta) : (!out, !out) -> !out
    %ca = tessera.call @su.ca(%c, %a) : (!c, !a) -> !a
    %atca = tessera.call @su.mm(%at, %ca) : (!at, !a) -> !out
    %result = tessera.call @su.add(%sum, %atca) : (!out, !out) -> !out
    llvm.return %result : !out
  }

  // A store between loads invalidates the repeated-variable match. CSE must
  // not treat the old and new contents as the same mathematical matrix.
  // REWRITE-LABEL: llvm.func @mutated
  // REWRITE: llvm.load %arg0
  // REWRITE: tessera.call @su.transpose
  // REWRITE: llvm.store %arg3, %arg0
  // REWRITE: llvm.load %arg0
  // REWRITE-NOT: tessera.guard
  // REWRITE-NOT: tessera.call @su.axpy
  // REWRITE-NOT: tessera.call @su.syr2k
  // REWRITE: llvm.return
  // LOWER-LABEL: llvm.func @mutated
  // LOWER-NOT: llvm.cond_br
  // LOWER-NOT: tessera.call @su.syr2k
  // LOWER: llvm.return
  llvm.func @mutated(%ap: !llvm.ptr, %b: !a, %c: !c {tessera.assumes = ["symmetric"]}, %new: !a) -> !out {
    %a0 = llvm.load %ap : !llvm.ptr -> !a
    %at = tessera.call @su.transpose(%a0) : (!a) -> !at
    %bt = tessera.call @su.transpose(%b) : (!a) -> !at
    %atb = tessera.call @su.mm(%at, %b) : (!at, !a) -> !out
    llvm.store %new, %ap : !a, !llvm.ptr
    %a1 = llvm.load %ap : !llvm.ptr -> !a
    %bta = tessera.call @su.mm(%bt, %a1) : (!at, !a) -> !out
    %sum = tessera.call @su.add(%atb, %bta) : (!out, !out) -> !out
    %ca = tessera.call @su.ca(%c, %a1) : (!c, !a) -> !a
    %atca = tessera.call @su.mm(%at, %ca) : (!at, !a) -> !out
    %result = tessera.call @su.add(%sum, %atca) : (!out, !out) -> !out
    llvm.return %result : !out
  }
}
