#ifndef ML_CODEGEN
#define ML_CODEGEN

#include <cassert>
#include <cstring>
#include <limits>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#include <ml_instr.h>
#include <ml_ssa_decl.h>

namespace ML {

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
void local_value_numbering(SSA& ssa){
  std::unordered_map<Instr, RegName, LocalValueNumberHash> vn;
  std::unordered_map<RegName, RegName, RegNameHash>        as;
  std::vector<Instr>                                       nins;
  for (auto& instr : ssa.instructions){
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
  ssa.instructions = nins;
}

RegSize estimate_register_size(SSA& ssa){
  RegSize largest = {0UL, 0UL};
  for (auto& instr : ssa.instructions){
    const SSAregData& ddat = ssa.context.lookup(instr.mDst);
    const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
    const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
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

std::vector<LiveSet> analyze_liveness(SSA& ssa){
  assert(ssa.instructions.size() < std::numeric_limits<long long>::max());
  long long size = static_cast<long long>(ssa.instructions.size());
  std::vector<LiveSet> res(ssa.instructions.size());
  bool is_changing = true;
  //NOTE: for a local block without branches, this should be done in 1 iteration
  while (is_changing){
    is_changing = false;
    for (long long i = size - 1; i >= 0; --i){
      LiveSet new_set;
      if (i == size - 1)
        new_set.liveout.insert(ssa.instructions[i].mDst); //assume the last temp should be live out
      else
        new_set.liveout = res[i+1].livein;
      new_set.livein = new_set.liveout;
      const SSAregData& dstDat = ssa.context.lookup(ssa.instructions[i].mDst);
      const SSAregData& s1Dat  = ssa.context.lookup(ssa.instructions[i].mSrc1);
      const SSAregData& s2Dat  = ssa.context.lookup(ssa.instructions[i].mSrc2);
      if (dstDat.mType == SSAregType::Mtx)
        new_set.livein.erase(ssa.instructions[i].mDst);
      if (s1Dat.mType == SSAregType::Mtx)
        new_set.livein.insert(ssa.instructions[i].mSrc1);
      if (s2Dat.mType == SSAregType::Mtx)
        new_set.livein.insert(ssa.instructions[i].mSrc2);
      if (new_set != res[i]){
        is_changing = true;
        res[i] = new_set;
      }
    }
  }
  return res;
}

//size_t estimate_gpu_local_registers(SSA& ssa, const std::vector<LiveSet>& live){
//  size_t res = 0;
//  for (size_t i = 0; i < ssa.instructions.size(); ++i){
//    size_t count = live[i].livein.size() + 1;
//    res = std::max(res, count);
//  }
//  return res;
//}

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

struct ComputeMtxCommunicator {
  static void clear_ssa(const Mtx& mtx){
    mtx.clear_ssa();
  }
  static double* get_data(const Mtx& mtx){
    return mtx.data();
  }
};

//TODO: do register allocation for GPU

} //ML

#endif//ML_CODEGEN
