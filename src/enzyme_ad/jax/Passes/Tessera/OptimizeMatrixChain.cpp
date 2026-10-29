//===----------------------------------------------------------------------===//
// Cost-based reassociation of explicitly opted-in dense matrix products.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::enzyme::tessera {
#define GEN_PASS_DEF_TESSERAOPTIMIZEMATRIXCHAINPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace mlir::enzyme::tessera

using namespace mlir;
using namespace mlir::enzyme::tessera;

namespace {

struct Matrix {
  uint64_t rows, cols;
  Type element;
};

// Descriptors are deliberately excluded: copying one does not snapshot its
// data.
std::optional<Matrix> matrixType(Type type) {
  llvm::SmallDenseSet<Type, 4> seen;
  while (auto record = dyn_cast_or_null<LLVM::LLVMStructType>(type)) {
    if (!seen.insert(type).second || record.isOpaque() ||
        record.getBody().size() != 1)
      return std::nullopt;
    type = record.getBody()[0];
  }
  auto rows = dyn_cast_or_null<LLVM::LLVMArrayType>(type);
  auto cols = rows ? dyn_cast<LLVM::LLVMArrayType>(rows.getElementType())
                   : LLVM::LLVMArrayType();
  if (!rows || !cols || !rows.getNumElements() || !cols.getNumElements() ||
      !isa<FloatType>(cols.getElementType()))
    return std::nullopt;
  return Matrix{rows.getNumElements(), cols.getNumElements(),
                cols.getElementType()};
}

struct Kernel {
  DefineOp define;
  Type lhs, rhs, result;
  uint64_t cost;
};

std::optional<Kernel> readKernel(DefineOp define) {
  auto results = define.getCallResultTypes();
  if (!define->getAttrOfType<UnitAttr>(kReassociateMatmulAttr) ||
      !define.getPure() || define.getNumCallOperands() != 2 ||
      results.size() != 1)
    return std::nullopt;
  Type lhs = define.getCallOperandValueType(0);
  Type rhs = define.getCallOperandValueType(1);
  Type result = results[0];
  auto a = matrixType(lhs), b = matrixType(rhs), c = matrixType(result);
  if (!a || !b || !c || a->cols != b->rows || c->rows != a->rows ||
      c->cols != b->cols || a->element != b->element ||
      a->element != c->element)
    return std::nullopt;
  uint64_t cost = llvm::SaturatingMultiply(
      llvm::SaturatingMultiply(a->rows, a->cols), b->cols);
  return Kernel{define, lhs, rhs, result, cost};
}

using Catalog = llvm::DenseMap<Operation *, const Kernel *>;
using KernelsByInputs =
    llvm::DenseMap<std::pair<Type, Type>, SmallVector<const Kernel *, 1>>;

const Kernel *lookupKernel(CallOp call, const Catalog &catalog) {
  auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
      call, call.getCalleeAttr());
  const Kernel *kernel = catalog.lookup(define.getOperation());
  if (!kernel || call.getNumOperands() != 2 || call.getNumResults() != 1 ||
      call.getOperand(0).getType() != kernel->lhs ||
      call.getOperand(1).getType() != kernel->rhs ||
      call.getResult(0).getType() != kernel->result)
    return nullptr;
  return kernel;
}

struct Chain {
  SmallVector<Value> leaves;
  SmallVector<CallOp> products;
  uint64_t cost = 0;
};

// shortcut: bound cubic planning to 64 leaves; revisit for longer real
// workloads.
constexpr unsigned kMaxLeaves = 64;

bool collectChain(Value value, CallOp root, const Catalog &catalog,
                  Chain &chain, unsigned depth = 0) {
  if (depth >= kMaxLeaves || chain.leaves.size() >= kMaxLeaves)
    return false;
  auto call = value.getDefiningOp<CallOp>();
  const Kernel *kernel = call ? lookupKernel(call, catalog) : nullptr;
  if (kernel && call->getBlock() == root->getBlock() &&
      (call == root || value.hasOneUse())) {
    for (Value operand : call.getOperands())
      if (!collectChain(operand, root, catalog, chain, depth + 1))
        return false;
    chain.products.push_back(call);
    chain.cost = llvm::SaturatingAdd(chain.cost, kernel->cost);
  } else {
    chain.leaves.push_back(value);
  }
  return true;
}

struct Plan {
  uint64_t cost;
  unsigned split = 0;
  const Kernel *kernel = nullptr;
};

// Keep a best plan per result type: equal shapes can have different C++ ABIs.
using Cell = llvm::MapVector<Type, Plan>;
using Table = SmallVector<SmallVector<Cell>>;

Value emitPlan(OpBuilder &builder, Location loc, const Chain &chain,
               const Table &table, unsigned first, unsigned last, Type type) {
  if (first == last)
    return chain.leaves[first];
  const Plan &plan = table[first][last].find(type)->second;
  Value lhs =
      emitPlan(builder, loc, chain, table, first, plan.split, plan.kernel->lhs);
  Value rhs = emitPlan(builder, loc, chain, table, plan.split + 1, last,
                       plan.kernel->rhs);
  DefineOp define = plan.kernel->define;
  auto call = CallOp::create(builder, loc, define.getSymNameAttr(),
                             TypeRange{type}, ValueRange{lhs, rhs});
  return call.getResult(0);
}

void optimizeChain(CallOp root, const KernelsByInputs &kernels,
                   const Catalog &catalog) {
  if (!lookupKernel(root, catalog) || root.getResult(0).use_empty())
    return;
  for (Operation *parent = root; parent; parent = parent->getParentOp())
    if (parent->hasAttr("tessera.no_rewrite"))
      return;

  // Plan a maximal chain once, rather than replanning each of its prefixes.
  if (root.getResult(0).hasOneUse()) {
    auto user = dyn_cast<CallOp>(*root.getResult(0).getUsers().begin());
    if (user && user->getBlock() == root->getBlock() &&
        lookupKernel(user, catalog))
      return;
  }

  Chain chain;
  if (!collectChain(root.getResult(0), root, catalog, chain) ||
      chain.leaves.size() < 3)
    return;
  unsigned count = chain.leaves.size();
  Table table(count);
  for (unsigned i = 0; i < count; ++i) {
    table[i].resize(count);
    table[i][i].try_emplace(chain.leaves[i].getType(), Plan{0});
  }
  for (unsigned length = 2; length <= count; ++length) {
    for (unsigned first = 0; first + length <= count; ++first) {
      unsigned last = first + length - 1;
      for (unsigned split = first; split < last; ++split) {
        for (const auto &[lhsType, lhs] : table[first][split])
          for (const auto &[rhsType, rhs] : table[split + 1][last]) {
            auto candidates = kernels.find({lhsType, rhsType});
            if (candidates == kernels.end())
              continue;
            for (const Kernel *kernel : candidates->second) {
              uint64_t cost =
                  llvm::SaturatingAdd(lhs.cost, rhs.cost, kernel->cost);
              // Saturated costs cannot distinguish competing parenthesizations.
              if (cost == std::numeric_limits<uint64_t>::max())
                continue;
              Cell &cell = table[first][last];
              auto existing = cell.find(kernel->result);
              if (existing == cell.end() || cost < existing->second.cost)
                cell[kernel->result] = Plan{cost, split, kernel};
            }
          }
      }
    }
  }
  auto best = table[0][count - 1].find(root.getResult(0).getType());
  if (best == table[0][count - 1].end() || best->second.cost >= chain.cost)
    return;

  root.emitRemark() << "reassociated matrix chain: scalar multiplication cost "
                    << chain.cost << " -> " << best->second.cost;
  OpBuilder builder(root);
  Value replacement = emitPlan(builder, root.getLoc(), chain, table, 0,
                               count - 1, root.getResult(0).getType());
  root.getResult(0).replaceAllUsesWith(replacement);
  for (CallOp call : llvm::reverse(chain.products))
    call.erase();
}

struct TesseraOptimizeMatrixChainPass
    : mlir::enzyme::tessera::impl::TesseraOptimizeMatrixChainPassBase<
          TesseraOptimizeMatrixChainPass> {
  using TesseraOptimizeMatrixChainPassBase::TesseraOptimizeMatrixChainPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Kernel> kernels;
    for (DefineOp define : module.getOps<DefineOp>()) {
      if (!define->hasAttr(kReassociateMatmulAttr))
        continue;
      if (auto kernel = readKernel(define))
        kernels.push_back(*kernel);
      else
        define.emitWarning() << "ignoring tessera.reassociate_matmul: expected "
                                "a pure binary product of compatible, nonempty "
                                "row-major floating-point matrix values";
    }
    if (kernels.empty())
      return;
    Catalog catalog;
    KernelsByInputs byInputs;
    for (Kernel &kernel : kernels) {
      catalog[kernel.define.getOperation()] = &kernel;
      byInputs[{kernel.lhs, kernel.rhs}].push_back(&kernel);
    }
    // Post-order visits producers first; rewrites erase only the current call
    // and preceding, single-use producers in its block.
    module.walk<WalkOrder::PostOrder>([&](CallOp call) {
      if (call->getParentOfType<ModuleOp>() == module)
        optimizeChain(call, byInputs, catalog);
    });
  }
};
} // namespace
