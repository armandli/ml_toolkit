#ifndef ML_SSA
#define ML_SSA

#include <cassert>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>

#include <cblas.h>

#include <ml_common.h>
#include <ml_exprtree.h>
#include <ml_instr.h>
#include <ml_matrix.h>
#include <ml_arena.h>
#include <ml_sse.h>

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

  std::unordered_map<const Mtx*, RegName>& get_mtxmap(){
    return mMtxMap;
  }
};

//TODO: hide private members
struct SSA {
  std::vector<Instr> instructions;
  SSAcontext         context;

  RegName merge(SSA& os, const Mtx& dmtx){
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

    std::unordered_map<const Mtx*, RegName>::iterator it = context.get_mtxmap().find(&dmtx);
    assert(it != context.get_mtxmap().end());
    return (*it).second;
  }

  void replace_old_ssa(std::shared_ptr<SSA> new_ssa){
    std::unordered_map<const Mtx*, RegName>& mtxes = context.get_mtxmap();
    for (std::unordered_map<const Mtx*, RegName>::iterator it = mtxes.begin(); it != mtxes.end(); ++it){
      const Mtx* mtx = (*it).first;
      if ((*mtx).is_ssa()){
        (*mtx).swap_ssa(new_ssa);
      }
    }
  }
};

//TODO: hide these recursion functions
template <typename CRTP> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<CRTP>& expr);

template <> RegName to_ssa(std::shared_ptr<SSA> ret, const MtxBase<MtxRef>& expr){
  const MtxRef& cexpr = static_cast<const MtxRef&>(expr);
  const Mtx& mtx = cexpr.mtx();
  //friend of Mtx class, update mtx's SSA instructions
  if (mtx.mSSA){
    RegName retname = (*ret).merge(*mtx.mSSA, mtx);
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

  assert(p1dat.mRows == p2dat.mRows || ((p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1)));

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

  assert(p1dat.mRows == p2dat.mRows || ((p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1)));

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

  assert(p1dat.mRows == p2dat.mRows || ((p1dat.mRows == 1 && p1dat.mCols == 1) || (p2dat.mRows == 1 && p2dat.mCols == 1)));

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

  (*ret).replace_old_ssa(ret);

  //friend of Mtx class, update destination matrix memory
  if (dst.mData != nullptr)
    delete[] dst.mData;
  dst.mRowStride = new_rowstride;
  dst.mColStride = new_colstride;
  dst.mData = new double[new_rowstride * new_colstride];
  dst.mRows = ddat.mRows;
  dst.mCols = ddat.mCols;

  (*ret).context.associate(dname, dst);
  return ret;
}

bool operator == (const Instr& a, const Instr& b){
  return a.mType == b.mType && a.mSrc1 == b.mSrc1 && a.mSrc2 == b.mSrc2;
}

struct LocalValueNumberHash {
  size_t operator()(const Instr& instr) const noexcept {
    unsigned long long ret = 0;
    char*              pr  = (char*)&ret;
    switch (instr.mType){
      case InstrType::Add:    pr[0] = 0; break;
      case InstrType::Sub:    pr[0] = 1; break;
      case InstrType::EMul:   pr[0] = 2; break;
      case InstrType::EDiv:   pr[0] = 3; break;
      case InstrType::Dot:    pr[0] = 4; break;
      case InstrType::AddMC:  pr[0] = 5; break;
      case InstrType::SubMC:  pr[0] = 6; break;
      case InstrType::SubCM:  pr[0] = 7; break;
      case InstrType::EMulMC: pr[0] = 8; break;
      case InstrType::EDivMC: pr[0] = 9; break;
      case InstrType::EDivCM: pr[0] = 10; break;
      case InstrType::Trn:    pr[0] = 11; break;
      default: assert(false);
    }
    memcpy(pr + 1, instr.mSrc1.name + 1, RegName::Len - 1);
    memcpy(pr + RegName::Len, instr.mSrc2.name + 1, RegName::Len - 1);

    return (size_t) ret; //DOES NOT WORK on 32bit systems!
  }
};

//benefits:
//  eliminate dead code
void local_value_numbering(std::shared_ptr<SSA> ssa){
  std::unordered_map<Instr, RegName, LocalValueNumberHash> vn;
  std::unordered_map<RegName, RegName, RegNameHash>        as;
  std::vector<Instr>                                       nins;
  for (auto& instr : (*ssa).instructions){
    Instr recon;
    recon.mType = instr.mType;
    decltype(as)::iterator rs1 = as.find(instr.mSrc1);
    decltype(as)::iterator rs2 = as.find(instr.mSrc2);
    if (rs1 != as.end()) recon.mSrc1 = (*rs1).second;
    else                 recon.mSrc1 = instr.mSrc1;
    if (rs2 != as.end()) recon.mSrc2 = (*rs2).second;
    else                 recon.mSrc2 = instr.mSrc2;

    decltype(vn)::iterator f = vn.find(recon);
    if (f == vn.end()){
      recon.mDst = instr.mDst;
      vn.insert(std::make_pair(recon, instr.mDst));
      nins.emplace_back(recon);
    } else
      as.insert(std::make_pair(instr.mDst, (*f).second));
  }
  (*ssa).instructions = nins;
}

RegSize estimate_register_size(std::shared_ptr<SSA> ssa){
  RegSize largest = {0UL, 0UL};
  for (auto& instr : (*ssa).instructions){
    const SSAregData& ddat = (*ssa).context.lookup(instr.mDst);
    const SSAregData& s1dat = (*ssa).context.lookup(instr.mSrc1);
    const SSAregData& s2dat = (*ssa).context.lookup(instr.mSrc2);
    largest.rs = std::max(largest.rs, ddat.mRows);
    largest.rs = std::max(largest.rs, s1dat.mRows);
    largest.rs = std::max(largest.rs, s2dat.mRows);
    largest.cs = std::max(largest.cs, ddat.mCols);
    largest.cs = std::max(largest.cs, s1dat.mCols);
    largest.cs = std::max(largest.cs, s2dat.mCols);
  }
  largest.rs = roundup_row(largest.rs);
  largest.cs = roundup_col(largest.cs);
  return largest;
}

struct LiveSet {
  std::unordered_set<RegName, RegNameHash> livein;
  std::unordered_set<RegName, RegNameHash> liveout;
};
bool operator==(const LiveSet& a, const LiveSet& b){
  return a.livein == b.livein && a.liveout == b.liveout;
}
bool operator!=(const LiveSet& a, const LiveSet& b){
  return not (a == b);
}

std::vector<LiveSet> analyze_liveness(std::shared_ptr<SSA> ssa){
  assert((*ssa).instructions.size() < std::numeric_limits<long long>::max());
  long long size = static_cast<long long>((*ssa).instructions.size());
  std::vector<LiveSet> res((*ssa).instructions.size());
  bool is_changing = true;
  //NOTE: for a local block without branches, this should be done in 1 iteration
  while (is_changing){
    is_changing = false;
    for (long long i = size - 1; i >= 0; --i){
      LiveSet new_set;
      if (i == size - 1)
        new_set.liveout.insert((*ssa).instructions[i].mDst); //assume the last temp should be live out
      else
        new_set.liveout = res[i+1].livein;
      new_set.livein = new_set.liveout;
      const SSAregData& dstDat = (*ssa).context.lookup((*ssa).instructions[i].mDst);
      const SSAregData& s1Dat  = (*ssa).context.lookup((*ssa).instructions[i].mSrc1);
      const SSAregData& s2Dat  = (*ssa).context.lookup((*ssa).instructions[i].mSrc2);
      if (dstDat.mType == SSAregType::Mtx)
        new_set.livein.erase((*ssa).instructions[i].mDst);
      if (s1Dat.mType == SSAregType::Mtx)
        new_set.livein.insert((*ssa).instructions[i].mSrc1);
      if (s2Dat.mType == SSAregType::Mtx)
        new_set.livein.insert((*ssa).instructions[i].mSrc2);
      if (new_set != res[i]){
        is_changing = true;
        res[i] = new_set;
      }
    }
  }
  return res;
}

//estimate the minimal number of temporaries needed for the block
size_t estimate_cpu_local_registers(std::shared_ptr<SSA> ssa, const std::vector<LiveSet>& live){
  size_t res = 0;
  for (size_t i = 0; i < (*ssa).instructions.size(); ++i){
    size_t count = live[i].livein.size();
    const SSAregData& dstData = (*ssa).context.lookup((*ssa).instructions[i].mDst);
    if (dstData.mType == SSAregType::Mtx && dstData.mMtxRef == nullptr)
      count++;
    for (auto name : live[i].livein){
      const SSAregData& dat = (*ssa).context.lookup(name);
      if (dat.mType == SSAregType::Mtx && dat.mMtxRef != nullptr)
        count--;
    }
    res = std::max(res, count);
  }
  return res;
}

size_t estimate_gpu_local_registers(std::shared_ptr<SSA> ssa, const std::vector<LiveSet>& live){
  size_t res = 0;
  for (size_t i = 0; i < (*ssa).instructions.size(); ++i){
    size_t count = live[i].livein.size() + 1;
    res = std::max(res, count);
  }
  return res;
}

enum class RegType : unsigned {
  Mem,
  Reg,
  Scl,
  Nil,
};

constexpr char MEM = 'm';
constexpr char REG = 'r';
constexpr char CST = 'c';
constexpr char NIL = 'n';

struct RegData {
  double* mMem;
  size_t  mRows;
  size_t  mCols;
  size_t  mRowStride;
  size_t  mColStride;

  RegData() = default;
  RegData(double* m, size_t r, size_t c):
    mMem(m), mRows(r), mCols(c), mRowStride(roundup_row(r)), mColStride(roundup_col(c)){}
};

class InstrContext {
  std::unordered_map<RegName, const Mtx*, RegNameHash> mMemMap;
  std::unordered_map<const Mtx*, RegName>              mMtxMap;
  std::unordered_map<RegName, RegData, RegNameHash>    mRegMap;
  std::unordered_map<RegName, double, RegNameHash>     mConstMap;
  std::unordered_map<double, RegName>                  mValMap;
  int                                                  mMemCount;
  int                                                  mRegCount;
  int                                                  mConstCount;

  RegName nextMemName(){
    RegName ret;
    sprintf(ret.name, "m%d", mMemCount++);
    return ret;
  }
  RegName nextRegName(){
    RegName ret;
    sprintf(ret.name, "r%d", mRegCount++);
    return ret;
  }
  RegName nextConstName(){
    RegName ret;
    sprintf(ret.name, "c%d", mConstCount++);
    return ret;
  }
public:
  InstrContext() = default;
  ~InstrContext() = default;

  RegType type(RegName name) const {
    switch (name.name[0]){
      case MEM: return RegType::Mem;
      case REG: return RegType::Reg;
      case CST: return RegType::Scl;
      case NIL: return RegType::Nil;
      default: assert(false);
    }
  }

  std::queue<RegName> gen_regs(){
    std::queue<RegName> ret;
    for (decltype(mRegMap)::iterator it = mRegMap.begin(); it != mRegMap.end(); ++it){
      ret.push((*it).first);
    }
    return ret;
  }

  void setRegSize(RegName name, size_t rows, size_t cols){
    assert(name.name[0] == REG);
    decltype(mRegMap)::iterator it = mRegMap.find(name);
    if (it != mRegMap.end()){
      RegData& dat = (*it).second;
      dat.mRows = rows;
      dat.mCols = cols;
      dat.mRowStride = roundup_row(rows);
      dat.mColStride = roundup_col(cols);
    }
  }

  RegName addReg(const RegData& data){
    RegName name = nextRegName();
    mRegMap.insert(std::make_pair(name, data));
    return name;
  }

  RegName addMem(const Mtx& mtx){
    decltype(mMtxMap)::iterator it = mMtxMap.find(&mtx);
    if (it == mMtxMap.end()){
      RegName name = nextMemName();
      mMemMap.insert(std::make_pair(name, &mtx));
      mMtxMap.insert(std::make_pair(&mtx, name));
      return name;
    } else
      return (*it).second;
  }

  RegName addConst(double val){
    decltype(mValMap)::iterator it = mValMap.find(val);
    if (it == mValMap.end()){
      RegName name = nextConstName();
      mConstMap.insert(std::make_pair(name, val));
      mValMap.insert(std::make_pair(val, name));
      return name;
    } else
      return (*it).second;
  }

  std::unordered_map<const Mtx*, RegName>&  memMap(){
    return mMtxMap;
  }

  const Mtx* lookup_mem(RegName name){
    assert(name.name[0] == MEM);
    return mMemMap[name];
  }

  RegData& lookup_reg(RegName name){
    assert(name.name[0] == REG);
    return mRegMap[name];
  }

  double lookup_val(RegName name){
    assert(name.name[0] == CST);
    return mConstMap[name];
  }
};

class MemInstrContext : public InstrContext {
  MemArena& mArena;
public:
  MemInstrContext(MemArena& arena, RegSize rsz, size_t reg_count): InstrContext(), mArena(arena) {
    for (size_t i = 0; i < reg_count; ++i)
      addReg(RegData(arena.reg(i), rsz.rs, rsz.cs));
  }
};

std::vector<Instr> local_register_allocation(std::shared_ptr<SSA> ssa, MemInstrContext& ctx, const std::vector<LiveSet>& liveness){
  std::vector<Instr> ret;
  std::unordered_map<RegName, RegName, RegNameHash> tmap;
  std::queue<RegName> fr = ctx.gen_regs();

  class GenName {
    std::unordered_map<RegName, RegName, RegNameHash>& tmap;
    std::queue<RegName>&                  fr;
    MemInstrContext&                      ctx;
  public:
    GenName(std::unordered_map<RegName, RegName, RegNameHash>& tmap, std::queue<RegName>& fr, MemInstrContext& ctx):
      tmap(tmap), fr(fr), ctx(ctx) {}
    RegName operator()(RegName sname, const SSAregData& sdat){
      RegName ret;
      switch (sdat.mType){
        case SSAregType::Mtx:
          if (sdat.mMtxRef != nullptr)
            ret = ctx.addMem(*sdat.mMtxRef);
          else {
            std::unordered_map<RegName, RegName, RegNameHash>::iterator it = tmap.find(sname);
            if (it == tmap.end()){
              assert(not fr.empty());
              ret = fr.front();
              fr.pop();
              tmap.insert(std::make_pair(sname, ret));
            } else {
              ret = (*it).second;
            }
          }
        break;
        case SSAregType::Scl:
          ret = ctx.addConst(sdat.mVal);
        break;
        case SSAregType::Nil:
          memcpy(ret.name, "nil", 3);
        break;
        default: assert(false);
      }
      return ret;
    }
  } gen_name(tmap, fr, ctx);

  for (size_t i = 0; i < (*ssa).instructions.size(); ++i){
    Instr& si = (*ssa).instructions[i];
    const SSAregData& s1dat = (*ssa).context.lookup(si.mSrc1);
    RegName r1name = gen_name(si.mSrc1, s1dat);
    const SSAregData& s2dat = (*ssa).context.lookup(si.mSrc2);
    RegName r2name = gen_name(si.mSrc2, s2dat);
    const SSAregData& ddat = (*ssa).context.lookup(si.mDst);
    RegName dname = gen_name(si.mDst, ddat);
    if (ctx.type(r1name) == RegType::Reg && liveness[i].liveout.find(si.mSrc1) == liveness[i].liveout.end()){
      fr.push(r1name);
      tmap.erase(si.mSrc1);
    }
    if (ctx.type(r2name) == RegType::Reg && liveness[i].liveout.find(si.mSrc2) == liveness[i].liveout.end()){
      fr.push(r2name);
      tmap.erase(si.mSrc2);
    }
    ret.emplace_back(Instr(si.mType, dname, r1name, r2name));
  }

  return ret;
}

void evaluate_cpu_instr(const std::vector<Instr>& instr, MemInstrContext& ctx){
  class MemFinder {
    MemInstrContext& ctx;
  public:
    MemFinder(MemInstrContext& ctx): ctx(ctx) {}
    double* operator()(RegName name){
      double* ret = nullptr;
      switch (ctx.type(name)){
        case RegType::Mem:
          ret = (*ctx.lookup_mem(name)).data();
        break;
        case RegType::Reg:
          ret = ctx.lookup_reg(name).mMem;
        break;
        case RegType::Scl:
        case RegType::Nil:
        break;
        default: assert(false);
      }
      return ret;
    }
  } find_mem(ctx);

  class SizeFinder {
    MemInstrContext& ctx;
  public:
    SizeFinder(MemInstrContext& ctx): ctx(ctx) {}
    RegSize operator()(RegName name){
      RegSize ret = {0, 0};
      switch (ctx.type(name)){
      case RegType::Mem: {
        const Mtx* m = ctx.lookup_mem(name);
        ret.rs = (*m).rows();
        ret.cs = (*m).cols();
      }
      break;
      case RegType::Reg: {
        RegData& dat = ctx.lookup_reg(name);
        ret.rs = dat.mRows;
        ret.cs = dat.mCols;
      }
      break;
      case RegType::Scl:
      case RegType::Nil:
      break;
      default: assert(false);
      }
      return ret;
    }
  } find_size(ctx);

  for (auto& si : instr){
    switch (si.mType){
      case InstrType::Add:
      case InstrType::Sub:
      case InstrType::EMul:
      case InstrType::EDiv: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr);
        assert(s2 != nullptr);
        assert(d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        assert(s1size.rs > 0 && s1size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.setRegSize(si.mDst, s1size.rs, s1size.cs);

        switch (si.mType){
          case InstrType::Add:  SSE::add_1d_sse_pd(d, s1, s2, roundup_row(s1size.rs), roundup_col(s1size.cs)); break;
          case InstrType::Sub:  SSE::sub_1d_sse_pd(d, s1, s2, roundup_row(s1size.rs), roundup_col(s1size.cs)); break;
          case InstrType::EMul: SSE::emul_1d_sse_pd(d, s1, s2, roundup_row(s1size.rs), roundup_col(s1size.cs)); break;
          case InstrType::EDiv: SSE::ediv_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::Dot: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr);
        assert(s2 != nullptr);
        assert(d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        RegSize s2size = find_size(si.mSrc2);
        assert(s1size.rs > 0 && s1size.cs > 0 && s2size.rs > 0 && s2size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.setRegSize(si.mDst, s1size.rs, s2size.cs);
        s1size.rs = roundup_row(s1size.rs);
        s1size.cs = roundup_col(s1size.cs);
        s2size.rs = roundup_row(s2size.rs);
        s2size.cs = roundup_row(s2size.cs);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    s1size.rs, s2size.cs, s1size.cs, 1,
                    s1, s1size.cs,
                    s2, s2size.cs,
                    0., d, s2size.cs);
      }
      break;
      case InstrType::SubMC:
      case InstrType::EDivMC: {
        double* s1 = find_mem(si.mSrc1);
        double  s2 = ctx.lookup_val(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr);
        assert(d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        assert(s1size.rs > 0 && s1size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.setRegSize(si.mDst, s1size.rs, s1size.cs);
        switch (si.mType){
          case InstrType::SubMC:  SSE::sub_mc_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::EDivMC: SSE::ediv_mc_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::SubCM:
      case InstrType::EDivCM: {
        double s1  = ctx.lookup_val(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s2 != nullptr);
        assert(d != nullptr);
        RegSize s2size = find_size(si.mSrc2);
        assert(s2size.rs > 0 && s2size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.setRegSize(si.mDst, s2size.rs, s2size.cs);
        switch (si.mType){
          case InstrType::SubCM:  SSE::sub_cm_2d_sse_pd(s1, d, s2, s2size.rs, s2size.cs, roundup_col(s2size.cs)); break;
          case InstrType::EDivCM: SSE::ediv_cm_2d_sse_pd(s1, d, s2, s2size.rs, s2size.cs, roundup_col(s2size.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::AddMC:
      case InstrType::EMulMC: {
        double  val = nan("");
        double* arg = nullptr;
        RegSize sz  = {0, 0};
        if (ctx.type(si.mSrc1) == RegType::Scl){
          val = ctx.lookup_val(si.mSrc1);
          arg = find_mem(si.mSrc2);
          sz  = find_size(si.mSrc2);
        } else {
          val = ctx.lookup_val(si.mSrc2);
          arg = find_mem(si.mSrc1);
          sz  = find_size(si.mSrc2);
        }
        double* d = find_mem(si.mDst);
        assert(arg != nullptr);
        assert(d != nullptr);
        switch (si.mType){
          case InstrType::AddMC:  SSE::add_const_2d_sse_pd(d, arg, val, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::EMulMC: SSE::emul_const_2d_sse_pd(d, arg, val, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::Trn: {
        double* s1 = find_mem(si.mSrc1);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr);
        assert(d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        assert(s1size.rs > 0 && s1size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.setRegSize(si.mDst, s1size.cs, s1size.rs);
        SSE::transpose4x4_2d_sse_pd(d, s1, roundup_row(s1size.rs), roundup_col(s1size.cs));
      }
      break;
      default: assert(false);
    }
  }
}

void release_ssa(MemInstrContext& ctx){
  std::unordered_map<const Mtx*, RegName>& outputs = ctx.memMap();
  for (std::unordered_map<const Mtx*, RegName>::iterator it = outputs.begin(); it != outputs.end(); ++it){
    const Mtx* pm = (*it).first;
    if ((*pm).mSSA)
      (*pm).mSSA.reset();
  }
}

void memvaluateSSA(std::shared_ptr<SSA> ssa, MemArena& arena){
  //GOTHERE
  std::cout << "Before Optimization: " << std::endl;
  std::cout << *ssa;

  //TODO: optimize given SSA: dead code elimination etc
  local_value_numbering(ssa);

  //GOTHERE
  std::cout << "After Optimization: " << std::endl;
  std::cout << *ssa;

  RegSize regsize = estimate_register_size(ssa);
  std::vector<LiveSet> liveness = analyze_liveness(ssa);
  size_t minregs = estimate_cpu_local_registers(ssa, liveness);
  arena.reset(regsize.rs * regsize.cs, minregs);
  MemInstrContext context(arena, regsize, minregs);

//  //GOTHERE
//  std::cout << "temps: " << minregs << std::endl;

  std::vector<Instr> instr = local_register_allocation(ssa, context, liveness);

  //GOTHERE
  std::cout << "Final Instructions:" << std::endl;
  for (size_t i = 0; i < instr.size(); ++i)
    std::cout << instr[i];

  evaluate_cpu_instr(instr, context);
  release_ssa(context);
}

//TODO: do register allocation for GPU


} //ML

#endif//ML_SSA
