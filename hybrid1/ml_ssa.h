#ifndef ML_SSA
#define ML_SSA

#include <cassert>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>

#include <ml_exprtree.h>
#include <ml_instr.h>
#include <ml_matrix.h>

namespace ML {

enum class SSAregType : unsigned {
  Scl,
  Mtx,
  Nil,
};

//TODO: bug: if a matrix takes an expression and reassign the result to itself, we break SSA rule
//TODO: currently the SSA are generated based on local block, NN typically include loops, maybe we should consider to moving to multi-block SSA
//TODO: hide private members of data structures from public use!!!

struct SSAregData {
  SSAregType mType;
  const Mtx* mMtxRef;
  size_t     mRows;
  size_t     mCols;
  double     mVal;

  SSAregData(const Mtx* ref, size_t rows, size_t cols):
    mType(SSAregType::Mtx), mMtxRef(ref), mRows(rows), mCols(cols), mVal(0){}
  explicit SSAregData(double v):
    mType(SSAregType::Scl), mMtxRef(nullptr), mRows(0), mCols(0), mVal(v){}
  SSAregData():
    mType(SSAregType::Nil), mMtxRef(nullptr), mRows(0), mCols(0), mVal(0.){}
};

class SSAcontext {
  std::unordered_map<const Mtx*, RegName>              mMtxMap;
  std::unordered_map<double, RegName>                  mConstMap;
  std::unordered_map<RegName, SSAregData, RegNameHash> mRegData;
  int                                                  mCounter;
  SSAregData                                           mNil;

  RegName nextName(){
    assert(mCounter < 1000);
    RegName ret;
    sprintf(ret.name, "s%d", mCounter++);
    return ret;
  }
public:
  SSAcontext(): mCounter(1), mNil(SSAregData()) {}

  RegName gen(const Mtx* mtx, size_t rows, size_t cols){
    RegName ret;
    if (mtx == nullptr){
      ret = nextName();
      mRegData.insert(std::make_pair(ret, SSAregData(nullptr, rows, cols)));
    } else if (mMtxMap.find(mtx) != mMtxMap.end()){
      memcpy(ret.name, mMtxMap[mtx].name, RegName::Len);
    } else {
      ret = nextName();
      mMtxMap.insert(std::make_pair(mtx, ret));
      mRegData.insert(std::make_pair(ret, SSAregData(mtx, rows, cols)));
    }
    return ret;
  }

  RegName gen(double v){
    decltype(mConstMap)::const_iterator it = mConstMap.find(v);
    RegName ret;
    if (it == mConstMap.end()){
      ret = nextName();
      mConstMap.insert(std::make_pair(v, ret));
      mRegData.insert(std::make_pair(ret, SSAregData(v)));
    } else {
      memcpy(ret.name, (*it).second.name, RegName::Len);
    }
    return ret;
  }

  RegName gen(){
    RegName nil;
    memcpy(nil.name, "s0\0\0", RegName::Len);
    return nil;
  }

  RegName gen(const SSAregData& dat){
    switch (dat.mType){
      case SSAregType::Mtx:
        return gen(dat.mMtxRef, dat.mRows, dat.mCols);
      case SSAregType::Scl:
        return gen(dat.mVal);
      case SSAregType::Nil:
        return gen();
      default: assert(false);
    }
  }

  const SSAregData& lookup(RegName name) const {
    if (strcmp(name.name, "s0") == 0)
      return mNil;

    decltype(mRegData)::const_iterator it = mRegData.find(name);
    if (it != mRegData.end())
      return (*it).second;
    assert(!!!"unknown register name lookup occurred. this should not happen");
  }

  void associate(RegName name, const Mtx& mtx){
    assert(mRegData.find(name) != mRegData.end());
    SSAregData& dat = mRegData[name];
    assert(dat.mType == SSAregType::Mtx);
    mMtxMap.insert(std::make_pair(&mtx, name));
    dat.mMtxRef = &mtx;
  }
};

//TODO: hide private members
struct SSA {
  std::vector<Instr> instructions;
  SSAcontext         context;

  RegName merge(SSA& os){
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
    } merge_name(merge_map, os.context, context);

    for (auto& instr : os.instructions){
      RegName new_s1 = merge_name(instr.mSrc1);
      RegName new_s2 = merge_name(instr.mSrc2);
      RegName new_d  = merge_name(instr.mDst);
      instructions.emplace_back(Instr(instr.mType, new_d, new_s1, new_s2));
    }
    RegName retname = instructions.back().mDst;
    return retname;
  }
};

//TODO: hide these recursion functions
template <typename CRTP> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<CRTP>& expr);

template <> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<MtxRef>& expr){
  const MtxRef& cexpr = static_cast<const MtxRef&>(expr);
  const Mtx& mtx = cexpr.mtx();
  //friend of Mtx class, update mtx's SSA instructions
  if (mtx.mSSA){
    RegName retname = (*ret).merge(*mtx.mSSA);
    mtx.swap_ssa(ret);
    return retname;
  } else
    return (*ret).context.gen(&mtx, mtx.rows(), mtx.cols());
}
template <> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<Scl>& expr){
  return (*ret).context.gen(static_cast<const Scl&>(expr).val());
}
template <typename X> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<Uop<TrnOp, X>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Uop<TrnOp, X>&>(expr).param());
  const SSAregData& p1dat = (*ret).context.lookup(p1);
  RegName p2;
  RegName dst = (*ret).context.gen(nullptr, p1dat.mCols, p1dat.mRows);
  (*ret).instructions.emplace_back(Instr(InstrType::Trn, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<Bop<AddOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<AddOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<AddOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = (*ret).context.lookup(p1);
  const SSAregData& p2dat = (*ret).context.lookup(p2);

  assert(p1dat.mRows == p2dat.mRows || ((p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1)));

  RegName dst = (*ret).context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl || p2dat.mType == SSAregType::Scl)
    (*ret).instructions.emplace_back(Instr(InstrType::AddMC, dst, p1, p2));
  else
    (*ret).instructions.emplace_back(Instr(InstrType::Add, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<Bop<SubOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<SubOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<SubOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = (*ret).context.lookup(p1);
  const SSAregData& p2dat = (*ret).context.lookup(p2);

  assert(p1dat.mRows == p2dat.mRows || (p1dat.mRows == 1 && p1dat.mCols == 1 || p2dat.mRows == 1 && p2dat.mCols == 1));

  RegName dst = (*ret).context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl)
    (*ret).instructions.emplace_back(Instr(InstrType::SubCM, dst, p1, p2));
  else if (p2dat.mType == SSAregType::Scl)
    (*ret).instructions.emplace_back(Instr(InstrType::SubMC, dst, p1, p2));
  else
    (*ret).instructions.emplace_back(Instr(InstrType::Sub, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<Bop<MulOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<MulOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<MulOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = (*ret).context.lookup(p1);
  const SSAregData& p2dat = (*ret).context.lookup(p2);

  assert(p1dat.mRows == p2dat.mRows || (p1dat.mRows == 1 && p1dat.mCols == 1 || p2dat.mRows == 1 && p2dat.mCols == 1));

  RegName dst = (*ret).context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl || p2dat.mType == SSAregType::Scl)
    (*ret).instructions.emplace_back(Instr(InstrType::EMulMC, dst, p1, p2));
  else
    (*ret).instructions.emplace_back(Instr(InstrType::EMul, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<Bop<DivOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<DivOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<DivOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = (*ret).context.lookup(p1);
  const SSAregData& p2dat = (*ret).context.lookup(p2);

  assert(p1dat.mRows == p2dat.mRows || (p1dat.mRows == 1 && p1dat.mCols == 1 || p2dat.mRows == 1 && p2dat.mCols == 1));

  RegName dst = (*ret).context.gen(nullptr, std::max(p1dat.mRows, p2dat.mRows), std::max(p1dat.mCols, p2dat.mCols));

  if (p1dat.mType == SSAregType::Scl)
    (*ret).instructions.emplace_back(Instr(InstrType::EDivCM, dst, p1, p2));
  else if (p2dat.mType == SSAregType::Scl)
    (*ret).instructions.emplace_back(Instr(InstrType::EDivMC, dst, p1, p2));
  else
    (*ret).instructions.emplace_back(Instr(InstrType::EDiv, dst, p1, p2));
  return dst;
}
template <typename X, typename Y> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<Bop<DotOp, X, Y>>& expr){
  RegName p1 = to_ssa(ret, static_cast<const Bop<DotOp, X, Y>&>(expr).param1());
  RegName p2 = to_ssa(ret, static_cast<const Bop<DotOp, X, Y>&>(expr).param2());
  const SSAregData& p1dat = (*ret).context.lookup(p1);
  const SSAregData& p2dat = (*ret).context.lookup(p2);

  assert(p1dat.mCols == p2dat.mRows);

  RegName dst = (*ret).context.gen(nullptr, p1dat.mRows, p2dat.mCols);
  (*ret).instructions.emplace_back(Instr(InstrType::Dot, dst, p1, p2));
  return dst;
}

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
      case InstrType::Trn:
        out << instr.mDst << " <- ~" << instr.mSrc1 << "\n";
      break;
      default: assert(false);
    }
  }
  return out;
}

template <typename CRTP>
std::shared_ptr<SSA> to_ssa(const MtxBase<CRTP>& expr, Mtx& dst){
  std::shared_ptr<SSA> ret(new SSA());
  RegName dname = to_ssa(ret, expr);
  const SSAregData& ddat = (*ret).context.lookup(dname);
  size_t new_rowstride = roundup_row(ddat.mRows);
  size_t new_colstride = roundup_col(ddat.mCols);

  //friend of Mtx class, update destination matrix memory
  if (dst.mData != nullptr && dst.mRowStride * dst.mColStride < new_rowstride * new_colstride){
    dst.mRowStride = new_rowstride;
    dst.mColStride = new_colstride;
    delete[] dst.mData;
    dst.mData = new double[new_rowstride * new_colstride];
  } else if (dst.mData == nullptr){
    dst.mRowStride = new_rowstride;
    dst.mColStride = new_colstride;
    dst.mData = new double[new_rowstride * new_colstride];
  }
  dst.mRows = ddat.mRows;
  dst.mCols = ddat.mCols;

  (*ret).context.associate(dname, dst);
  return ret;
}

} //ML

#endif//ML_SSA
