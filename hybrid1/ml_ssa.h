#ifndef ML_SSA
#define ML_SSA

#include <cassert>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <ostream>

#include <cblas.h>

#include <ml_common.h>
#include <ml_exprtree.h>
#include <ml_instr.h>
#include <ml_ssa_decl.h>
#include <ml_matrix.h>

namespace ML {

//TODO: bug: if a matrix takes an expression and reassign the result to itself, we break SSA rule
//TODO: currently the SSA are generated based on local block, NN typically include loops, maybe we should consider to moving to multi-block SSA
//TODO: hide private members of data structures from public use!!!

struct SSAMtxCommunicator {
  static void reset_mtx_size(Mtx& dst, const SSAregData& data){
    size_t rowstride = roundup_row(data.mRows);
    size_t colstride = roundup_col(data.mCols);
    if (dst.mData != nullptr)
      delete[] dst.mData;
    dst.mRowStride = rowstride;
    dst.mColStride = colstride;
    dst.mData = new double[rowstride * colstride];
    dst.mRows = data.mRows;
    dst.mCols = data.mCols;
  }
  static RegName merge_instructions(SSA& dst, const Mtx& src){
    //if there is no SSABlock in src, we're done
    if (not src.is_ssa()) return dst.context.gen(&src, src.rows(), src.cols());

    SSA& src_ssa = src.ssa();
    std::unordered_map<RegName, RegName, RegNameHash> merge_map;

    class RegNameMerger {
      std::unordered_map<RegName, RegName, RegNameHash>& merge_map;
      SSAcontext&                                        old_context;
      SSAcontext&                                        new_context;
    public:
      RegNameMerger(std::unordered_map<RegName, RegName, RegNameHash>& mm, SSAcontext& oc, SSAcontext& nc):
        merge_map(mm), old_context(oc), new_context(nc) {}
      RegName operator()(RegName old_name){
        RegName ret;
        std::unordered_map<RegName, RegName, RegNameHash>::iterator it = merge_map.find(old_name);
        if (it == merge_map.end()){
          const SSAregData& dat = old_context.lookup(old_name);
          ret = new_context.gen(dat);
          merge_map.insert(std::make_pair(old_name, ret));
        } else {
          ret = (*it).second;
        }
        return ret;
      }
    } merge_name(merge_map, src_ssa.context, dst.context);

    for (auto& instr : src_ssa.instructions){
      RegName new_s1 = merge_name(instr.mSrc1);
      RegName new_s2 = merge_name(instr.mSrc2);
      RegName new_d  = merge_name(instr.mDst);
      dst.instructions.emplace_back(Instr(instr.mType, new_d, new_s1, new_s2));
    }

    std::unordered_map<const Mtx*, RegName>::iterator it = dst.context.get_mtxmap().find(&src);
    assert(it != dst.context.get_mtxmap().end());
    return (*it).second;
  }
};

//TODO: hide these recursion functions
template <typename CRTP> RegName to_ssa(SSA&, const MtxBase<CRTP>&);

template <> RegName to_ssa(SSA& ret, const MtxBase<MtxRef>& expr){
  const MtxRef& cexpr = static_cast<const MtxRef&>(expr);
  const Mtx& mtx = cexpr.mtx();
  return SSAMtxCommunicator::merge_instructions(ret, mtx);
}
template <> RegName to_ssa(SSA& ret, const MtxBase<Scl>& expr){
  return ret.context.gen(static_cast<const Scl&>(expr).val());
}
template <typename X> RegName to_ssa(SSA& ret, const MtxBase<Uop<TrnOp, X>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Uop<TrnOp, X>&>(expr).param());
  const SSAregData& p1dat = ret.context.lookup(p1);
  RegName p2;
  RegName dst = ret.context.gen(nullptr, p1dat.mCols, p1dat.mRows);
  ret.instructions.emplace_back(Instr(InstrType::Trn, dst, p1, p2));
  return dst;
}
template <typename X> RegName to_ssa(SSA& ret, const MtxBase<Uop<NotOp, X>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Uop<NotOp, X>&>(expr).param());
  const SSAregData& p1dat = ret.context.lookup(p1);
  RegName p2;
  RegName dst = ret.context.gen(nullptr, p1dat.mRows, p1dat.mCols);
  ret.instructions.emplace_back(Instr(InstrType::Not, dst, p1, p2));
  return dst;
}
template <typename X> RegName to_ssa(SSA& ret, const MtxBase<Uop<TanhOp, X>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Uop<TanhOp, X>&>(expr).param());
  const SSAregData& p1dat = ret.context.lookup(p1);
  RegName p2;
  RegName dst = ret.context.gen(nullptr, p1dat.mRows, p1dat.mCols);
  ret.instructions.emplace_back(Instr(InstrType::Tanh, dst, p1, p2));
  return dst;
}
template <typename X> RegName to_ssa(SSA& ret, const MtxBase<Uop<SoftmaxOp, X>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Uop<SoftmaxOp, X>&>(expr).param());
  const SSAregData& p1dat = ret.context.lookup(p1);
  RegName p2;
  RegName dst = ret.context.gen(nullptr, p1dat.mRows, p1dat.mCols);
  ret.instructions.emplace_back(Instr(InstrType::Softmax, dst, p1, p2));
  return dst;
}
template <typename X> RegName to_ssa(SSA& ret, const MtxBase<Uop<ExpOp, X>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Uop<ExpOp, X>&>(expr).param());
  const SSAregData& p1dat = ret.context.lookup(p1);
  RegName p2;
  RegName dst = ret.context.gen(nullptr, p1dat.mRows, p1dat.mCols);
  ret.instructions.emplace_back(Instr(InstrType::Exp, dst, p1, p2));
  return dst;
}
template <typename X> RegName to_ssa(SSA& ret, const MtxBase<Uop<IsnanOp, X>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Uop<IsnanOp, X>&>(expr).param());
  const SSAregData& p1dat = ret.context.lookup(p1);
  RegName p2;
  RegName dst = ret.context.gen(nullptr, p1dat.mRows, p1dat.mCols);
  ret.instructions.emplace_back(Instr(InstrType::Isnan, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(SSA& ret, const MtxBase<Bop<AddOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<AddOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<AddOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = ret.context.lookup(p1);
  const SSAregData& p2dat = ret.context.lookup(p2);

  assert((p1dat.mRows == p2dat.mRows && p1dat.mCols == p2dat.mCols) || ((p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1)));

  RegName dst = ret.context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl || p2dat.mType == SSAregType::Scl)
    ret.instructions.emplace_back(Instr(InstrType::AddMC, dst, p1, p2));
  else
    ret.instructions.emplace_back(Instr(InstrType::Add, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(SSA& ret, const MtxBase<Bop<SubOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<SubOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<SubOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = ret.context.lookup(p1);
  const SSAregData& p2dat = ret.context.lookup(p2);

  assert((p1dat.mRows == p2dat.mRows && p1dat.mCols == p2dat.mCols) || ((p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1)));

  RegName dst = ret.context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl)
    ret.instructions.emplace_back(Instr(InstrType::SubCM, dst, p1, p2));
  else if (p2dat.mType == SSAregType::Scl)
    ret.instructions.emplace_back(Instr(InstrType::SubMC, dst, p1, p2));
  else
    ret.instructions.emplace_back(Instr(InstrType::Sub, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(SSA& ret, const MtxBase<Bop<MulOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<MulOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<MulOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = ret.context.lookup(p1);
  const SSAregData& p2dat = ret.context.lookup(p2);

  assert((p1dat.mRows == p2dat.mRows && p1dat.mCols == p2dat.mCols) || ((p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1)));

  RegName dst = ret.context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl || p2dat.mType == SSAregType::Scl)
    ret.instructions.emplace_back(Instr(InstrType::EMulMC, dst, p1, p2));
  else
    ret.instructions.emplace_back(Instr(InstrType::EMul, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(SSA& ret, const MtxBase<Bop<DivOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<DivOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<DivOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = ret.context.lookup(p1);
  const SSAregData& p2dat = ret.context.lookup(p2);

  assert((p1dat.mRows == p2dat.mRows && p1dat.mCols == p2dat.mCols) || ((p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1)));

  RegName dst = ret.context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl)
    ret.instructions.emplace_back(Instr(InstrType::EDivCM, dst, p1, p2));
  else if (p2dat.mType == SSAregType::Scl)
    ret.instructions.emplace_back(Instr(InstrType::EDivMC, dst, p1, p2));
  else
    ret.instructions.emplace_back(Instr(InstrType::EDiv, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(SSA& ret, const MtxBase<Bop<DotOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<DotOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<DotOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = ret.context.lookup(p1);
  const SSAregData& p2dat = ret.context.lookup(p2);

  assert(p1dat.mCols == p2dat.mRows);

  RegName dst = ret.context.gen(nullptr, p1dat.mRows, p2dat.mCols);
  ret.instructions.emplace_back(Instr(InstrType::Dot, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(SSA& ret, const MtxBase<Bop<GtOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<GtOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<GtOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = ret.context.lookup(p1);
  const SSAregData& p2dat = ret.context.lookup(p2);

  assert((p1dat.mRows == p2dat.mRows && p1dat.mCols == p2dat.mCols) || (p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1));

  RegName dst = ret.context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl)
    ret.instructions.emplace_back(Instr(InstrType::GTCM, dst, p1, p2));
  else if (p2dat.mType == SSAregType::Scl)
    ret.instructions.emplace_back(Instr(InstrType::GTMC, dst, p1, p2));
  else
    ret.instructions.emplace_back(Instr(InstrType::GT, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(SSA& ret, const MtxBase<Bop<MaskOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<MaskOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<MaskOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = ret.context.lookup(p1);
  const SSAregData& p2dat = ret.context.lookup(p2);

  assert(p1dat.mRows == p2dat.mRows && p1dat.mCols == p2dat.mCols);

  RegName dst = ret.context.gen(nullptr, p1dat.mRows, p1dat.mCols);
  ret.instructions.emplace_back(Instr(InstrType::Mask, dst, p1, p2));
  return dst;
}
//TODO: expand operation here

std::ostream& operator << (std::ostream& out, const SSA& ssa){
  for (auto& instr : ssa.instructions){
    switch (instr.mType){
      case InstrType::Add:
        out << instr.mDst << " <- " << instr.mSrc1 << " + " << instr.mSrc2 << "\n";
      break;
      case InstrType::Sub:
        out << instr.mDst << " <- " << instr.mSrc1 << " - " << instr.mSrc2 << "\n";
      break;
      case InstrType::EMul:
        out << instr.mDst << " <- " << instr.mSrc1 << " * " << instr.mSrc2 << "\n";
      break;
      case InstrType::EDiv:
        out << instr.mDst << " <- " << instr.mSrc1 << " / " << instr.mSrc2 << "\n";
      break;
      case InstrType::Dot:
        out << instr.mDst << " <- " << instr.mSrc1 << " ^ " << instr.mSrc2 << "\n";
      break;
      case InstrType::Mask:
        out << instr.mDst << " <- " << instr.mSrc1 << " & " << instr.mSrc2 << "\n";
      break;
      case InstrType::GT:
        out << instr.mDst << " <- " << instr.mSrc1 << " > " << instr.mSrc2 << "\n";
      break;
      case InstrType::AddMC: {
        const SSAregData& arg1 = ssa.context.lookup(instr.mSrc1);
        const SSAregData& arg2 = ssa.context.lookup(instr.mSrc2);
        out << instr.mDst << " <- ";
        switch (arg1.mType){
          case SSAregType::Scl: out << arg1.mVal; break;
          case SSAregType::Mtx: out << instr.mSrc1; break;
          default:              assert(false);
        }
        out << " + ";
        switch (arg2.mType){
          case SSAregType::Scl: out << arg2.mVal; break;
          case SSAregType::Mtx: out << instr.mSrc1; break;
          default:              assert(false);
        }
        out << "\n";
      }
      break;
      case InstrType::SubMC:
        out << instr.mDst << " <- " << instr.mSrc1 << " - " << ssa.context.lookup(instr.mSrc2).mVal << "\n";
      break;
      case InstrType::SubCM:
        out << instr.mDst << " <- " << ssa.context.lookup(instr.mSrc1).mVal << " - " << instr.mSrc2 << "\n";
      break;
      case InstrType::EMulMC: {
        const SSAregData& arg1 = ssa.context.lookup(instr.mSrc1);
        const SSAregData& arg2 = ssa.context.lookup(instr.mSrc2);
        out << instr.mDst << " <- "; 
        switch (arg1.mType){
          case SSAregType::Scl: out << arg1.mVal; break;
          case SSAregType::Mtx: out << instr.mSrc1; break;
          default:              assert(false);
        }
        out << " * ";
        switch (arg2.mType){
          case SSAregType::Scl: out << arg2.mVal; break;
          case SSAregType::Mtx: out << instr.mSrc1; break;
          default:              assert(false);
        }
        out << "\n";
      }
      break;
      case InstrType::EDivMC:
        out << instr.mDst << " <- " << instr.mSrc1 << " / " << ssa.context.lookup(instr.mSrc2).mVal << "\n";
      break;
      case InstrType::EDivCM:
        out << instr.mDst << " <- " << ssa.context.lookup(instr.mSrc1).mVal << " / " << instr.mSrc2 << "\n";
      break;
      case InstrType::GTMC:
        out << instr.mDst << " <- " << instr.mSrc1 << " > " << ssa.context.lookup(instr.mSrc2).mVal << "\n";
      break;
      case InstrType::GTCM:
        out << instr.mDst << " <- " << ssa.context.lookup(instr.mSrc1).mVal << " > " << instr.mSrc2 << "\n";
      break;
      case InstrType::Trn:
        out << instr.mDst << " <- ~" << instr.mSrc1 << "\n";
      break;
      case InstrType::Not:
        out << instr.mDst << " <- ! " << instr.mSrc1 << "\n";
      break;
      case InstrType::Tanh:
        out << instr.mDst << " <- tanh(" << instr.mSrc1 << ")\n";
      break;
      case InstrType::Softmax:
        out << instr.mDst << " <- softmax(" << instr.mSrc1 << ")\n";
      break;
      case InstrType::Exp:
        out << instr.mDst << " <- exp(" << instr.mSrc1 << ")\n";
      break;
      case InstrType::Isnan:
        out << instr.mDst << " <- isnan(" << instr.mSrc1 << ")\n";
      break;
      //TODO: expand operation here
      default: assert(false);
    }
  }
  return out;
}

template <typename CRTP>
SSA to_ssa(const MtxBase<CRTP>& expr, Mtx& dst){
  SSA ret;
  RegName dname = to_ssa(ret, expr);
  const SSAregData& ddat = ret.context.lookup(dname);
  SSAMtxCommunicator::reset_mtx_size(dst, ddat);
  ret.context.associate(dname, dst);
  return ret;
}

} //ML

#endif//ML_SSA
