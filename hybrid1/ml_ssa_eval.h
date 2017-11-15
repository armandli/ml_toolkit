#ifndef ML_SSA_EVAL
#define ML_SSA_EVAL

#include <memory>
#include <vector>
#include <queue>
#include <algorithm>

#include <cblas.h>

#include <ml_common.h>
#include <ml_instr.h>
#include <ml_ssa.h>
#include <ml_arena.h>
#include <ml_sse.h>

namespace ML {

//TODO: do local instruction consolidation on SSA, this includes dead code elimination

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
  std::vector<LiveSet> res((*ssa).instructions.size());
  bool is_changing = true;
  //NOTE: for a local block without branches, this should be done in 1 iteration
  while (is_changing){
    is_changing = false;
    for (int i = (*ssa).instructions.size() - 1; i >= 0; --i){
      LiveSet new_set;
      if (i == (*ssa).instructions.size() - 1)
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
//  //GOTHERE
//  std::cout << *ssa;

  //TODO: optimize given SSA: dead code elimination etc

  RegSize regsize = estimate_register_size(ssa);
  std::vector<LiveSet> liveness = analyze_liveness(ssa);
  size_t minregs = estimate_cpu_local_registers(ssa, liveness);
  arena.reset(regsize.rs * regsize.cs, minregs);
  MemInstrContext context(arena, regsize, minregs);

//  //GOTHERE
//  std::cout << "temps: " << minregs << std::endl;

  std::vector<Instr> instr = local_register_allocation(ssa, context, liveness);

//  //GOTHERE
//  for (size_t i = 0; i < instr.size(); ++i)
//    std::cout << instr[i];

  evaluate_cpu_instr(instr, context);
  release_ssa(context);
}

//TODO: do register allocation for GPU

}

#endif//ML_SSA_EVAL
