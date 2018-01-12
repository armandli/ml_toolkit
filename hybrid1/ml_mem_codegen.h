#ifndef ML_MEM_CODEGEN
#define ML_MEM_CODEGEN

#include <vector>
#include <unordered_map>
#include <queue>

#include <cblas.h>

#include <ml_instr.h>
#include <ml_ssa_decl.h>
#include <ml_codegen.h>
#include <ml_arena.h>

//TODO: have to take care of what happens if memory is not fresh!!! especially arena memory is not always cleared

namespace ML {

//estimate the minimal number of temporaries needed for the block
size_t estimate_cpu_local_registers(SSA& ssa, const std::vector<LiveSet>& live){
  size_t res = 0;
  for (size_t i = 0; i < ssa.instructions.size(); ++i){
    size_t count = live[i].livein.size();
    const SSAregData& dstData = ssa.context.lookup(ssa.instructions[i].mDst);
    if (dstData.mType == SSAregType::Mtx && dstData.mMtxRef == nullptr)
      count++;
    for (auto name : live[i].livein){
      const SSAregData& dat = ssa.context.lookup(name);
      if (dat.mType == SSAregType::Mtx && dat.mMtxRef != nullptr)
        count--;
    }
    res = std::max(res, count);
  }
  return res;
}

class MemInstrContext : public InstrContext {
  MemArena& mArena;
public:
  MemInstrContext(MemArena& arena, RegSize rsz, size_t reg_count): InstrContext(), mArena(arena) {
    for (size_t i = 0; i < reg_count; ++i)
      add_reg(RegData(arena.reg(i), rsz.rs, rsz.cs));
  }

  std::queue<RegName> gen_regs(){
    std::queue<RegName> ret;
    for (decltype(mRegMap)::iterator it = mRegMap.begin(); it != mRegMap.end(); ++it){
      ret.push((*it).first);
    }
    return ret;
  }
};

std::vector<Instr> local_register_allocation(SSA& ssa, MemInstrContext& ctx, const std::vector<LiveSet>& liveness){
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
            ret = ctx.add_mem(*sdat.mMtxRef);
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
          ret = ctx.add_const(sdat.mVal);
        break;
        case SSAregType::Nil:
          ret = ctx.add_nil();
        break;
        default: assert(false);
      }
      return ret;
    }
  } gen_name(tmap, fr, ctx);

  for (size_t i = 0; i < ssa.instructions.size(); ++i){
    Instr& si = ssa.instructions[i];
    const SSAregData& s1dat = ssa.context.lookup(si.mSrc1);
    RegName r1name = gen_name(si.mSrc1, s1dat);
    const SSAregData& s2dat = ssa.context.lookup(si.mSrc2);
    RegName r2name = gen_name(si.mSrc2, s2dat);
    const SSAregData& ddat = ssa.context.lookup(si.mDst);
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
          ret = ComputeMtxCommunicator::get_data((*ctx.lookup_mem(name)));
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
        const Memory* m = ctx.lookup_mem(name);
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
      case InstrType::EDiv:
      case InstrType::GT:
      case InstrType::Mask:
      case InstrType::DRelu:
      case InstrType::DSigmoid:
      case InstrType::DTanh:
      case InstrType::Deriviative:
      case InstrType::DSS:
      case InstrType::MSELoss:
      case InstrType::MSEAccuracy: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && s2 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        assert(s1size.rs > 0 && s1size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, s1size.rs, s1size.cs);
        switch (si.mType){
          case InstrType::Add:         SSE::add_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::Sub:         SSE::sub_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::EMul:        SSE::emul_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::EDiv:        SSE::ediv_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::GT:          SSE::gt_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::Mask:        SSE::mask_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::DRelu:       SSE::drelu_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::DSigmoid:    SSE::dsigmoid_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::DTanh:       SSE::dtanh_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::Deriviative: SSE::deriviative_row_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::DSS:         SSE::diff_square_sum_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::MSELoss:     SSE::mse_loss_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::MSEAccuracy: SSE::mse_accuracy_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::Dot: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && s2 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        RegSize s2size = find_size(si.mSrc2);
        assert(s1size.rs > 0 && s1size.cs > 0 && s2size.rs > 0 && s2size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, s1size.rs, s2size.cs);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    s1size.rs, s2size.cs, s1size.cs, 1,
                    s1, roundup_col(s1size.cs),
                    s2, roundup_col(s2size.cs),
                    0., d, roundup_col(s2size.cs));
      }
      break;
      case InstrType::Trn1Dot: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && s2 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        RegSize s2size = find_size(si.mSrc2);
        assert(s1size.rs > 0 && s1size.cs > 0 && s2size.rs > 0 && s2size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, s1size.cs, s2size.cs);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    s1size.cs, s2size.cs, s2size.rs, 1,
                    s1, roundup_col(s1size.cs),
                    s2, roundup_col(s2size.cs),
                    0., d, roundup_col(s2size.cs));
      }
      break;
      case InstrType::Trn2Dot: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && s2 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        RegSize s2size = find_size(si.mSrc2);
        assert(s1size.rs > 0 && s1size.cs > 0 && s2size.rs > 0 && s2size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, s1size.rs, s2size.rs);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    s1size.rs, s2size.rs, s1size.cs, 1,
                    s1, roundup_col(s1size.cs),
                    s2, roundup_col(s2size.cs),
                    0., d, roundup_col(s2size.rs));
      }
      break;
      case InstrType::Trn3Dot: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && s2 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        RegSize s2size = find_size(si.mSrc2);
        assert(s1size.rs > 0 && s1size.cs > 0 && s2size.rs > 0 && s2size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, s1size.cs, s2size.rs);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                    s1size.cs, s2size.rs, s1size.rs, 1,
                    s1, roundup_col(s1size.cs),
                    s2, roundup_col(s2size.cs),
                    0., d, roundup_col(s2size.rs));
      }
      break;
      case InstrType::CELoss:
      case InstrType::CEAccuracy: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && s2 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        assert(s1size.rs > 0 && s1size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, 1, 1);
        switch (si.mType){
          case InstrType::CELoss:     MTXOP::ce_loss_2d_mtxop_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::CEAccuracy: MTXOP::ce_accuracy_2d_mtxop_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::SubMC:
      case InstrType::EDivMC:
      case InstrType::GTMC:
      case InstrType::GT0MC: {
        double* s1 = find_mem(si.mSrc1);
        double  s2 = ctx.lookup_val(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        assert(s1size.rs > 0 && s1size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, s1size.rs, s1size.cs);
        switch (si.mType){
          case InstrType::SubMC:  SSE::sub_mc_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::EDivMC: SSE::ediv_mc_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::GTMC:   SSE::gt_mc_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          case InstrType::GT0MC:  SSE::gt0_mc_2d_sse_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_col(s1size.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::SubCM:
      case InstrType::EDivCM:
      case InstrType::GTCM:
      case InstrType::GT0CM: {
        double s1  = ctx.lookup_val(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s2 != nullptr && d != nullptr);
        RegSize s2size = find_size(si.mSrc2);
        assert(s2size.rs > 0 && s2size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, s2size.rs, s2size.cs);
        switch (si.mType){
          case InstrType::SubCM:  SSE::sub_cm_2d_sse_pd(s1, d, s2, s2size.rs, s2size.cs, roundup_col(s2size.cs)); break;
          case InstrType::EDivCM: SSE::ediv_cm_2d_sse_pd(s1, d, s2, s2size.rs, s2size.cs, roundup_col(s2size.cs)); break;
          case InstrType::GTCM:   SSE::gt_cm_2d_sse_pd(s1, d, s2, s2size.rs, s2size.cs, roundup_col(s2size.cs)); break;
          case InstrType::GT0CM:  SSE::gt0_cm_2d_sse_pd(s1, s2, s2, s2size.rs, s2size.cs, roundup_col(s2size.cs)); break;
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
          sz  = find_size(si.mSrc1);
        }
        double* d = find_mem(si.mDst);
        assert(arg != nullptr && d != nullptr);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, sz.rs, sz.cs);
        switch (si.mType){
          case InstrType::AddMC:  SSE::add_const_2d_sse_pd(d, arg, val, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::EMulMC: SSE::emul_const_2d_sse_pd(d, arg, val, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::AddCC:
      case InstrType::SubCC:
      case InstrType::EMulCC:
      case InstrType::EDivCC: {
        double s1, s2;
        if (ctx.type(si.mSrc1) == RegType::Scl)
          s1 = ctx.lookup_val(si.mSrc1);
        else
          s1 = *find_mem(si.mSrc1);
        if (ctx.type(si.mSrc2) == RegType::Scl)
          s2 = ctx.lookup_val(si.mSrc2);
        else
          s2 = *find_mem(si.mSrc2);
        double* d = find_mem(si.mDst);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, 1, 1);
        switch (si.mType){
          case InstrType::AddCC:  MTXOP::add_cc_1d_mtxop_pd(d, s1, s2); break;
          case InstrType::SubCC:  MTXOP::sub_cc_1d_mtxop_pd(d, s1, s2); break;
          case InstrType::EMulCC: MTXOP::emul_cc_1d_mtxop_pd(d, s1, s2); break;
          case InstrType::EDivCC: MTXOP::ediv_cc_1d_mtxop_pd(d, s1, s2); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::L2Loss: {
        double val = nan("");
        double* arg = nullptr;
        RegSize sz = {0, 0};
        if (ctx.type(si.mSrc1) == RegType::Scl){
          val = ctx.lookup_val(si.mSrc1);
          arg = find_mem(si.mSrc2);
          sz  = find_size(si.mSrc2);
        } else {
          val = ctx.lookup_val(si.mSrc2);
          arg = find_mem(si.mSrc1);
          sz  = find_size(si.mSrc1);
        }
        double* d = find_mem(si.mDst);
        assert(arg != nullptr && d != nullptr);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, 1, 1);
        SSE::loss_l2_2d_sse_pd(d, arg, val, sz.rs, sz.cs, roundup_col(sz.cs)); break;
      }
      break;
      case InstrType::Trn: {
        double* s1 = find_mem(si.mSrc1);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        assert(s1size.rs > 0 && s1size.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, s1size.cs, s1size.rs);
        SSE::transpose4x4_2d_sse_pd(d, s1, roundup_row(s1size.cs), roundup_row(s1size.rs), roundup_col(s1size.cs), roundup_col(s1size.rs));
      }
      break;
      case InstrType::Copy:
      case InstrType::Tanh:
      case InstrType::Softmax:
      case InstrType::Exp:
      case InstrType::Not:
      case InstrType::Isnan:
      case InstrType::Isnan0:
      case InstrType::Sqrt:
      case InstrType::Abs:
      case InstrType::Sigmoid: {
        double* s1 = find_mem(si.mSrc1);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && d != nullptr);
        RegSize sz = find_size(si.mSrc1);
        assert(sz.rs > 0 && sz.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, sz.rs, sz.cs);
        switch (si.mType){
          case InstrType::Copy:    SSE::copy_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Tanh:    SSE::tanh_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Softmax: SSE::softmax_r_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Exp:     SSE::exp_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Not:     SSE::not_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Isnan:   SSE::isnan_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Isnan0:  SSE::isnan0_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Sqrt:    SSE::sqrt_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Abs:     MTXOP::abs_2d_mtxop_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          case InstrType::Sigmoid: SSE::sigmoid_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::Sum: {
        double* s1 = find_mem(si.mSrc1);
        double* d = find_mem(si.mDst);
        assert(s1 != nullptr && d != nullptr);
        RegSize sz = find_size(si.mSrc1);
        assert(sz.rs > 0 && sz.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, 1, 1);
        SSE::sum_all_2d_sse_pd(d, s1, sz.rs, sz.cs, roundup_col(sz.cs));
      }
      break;
      case InstrType::SqrtC: {
        double src;
        if (ctx.type(si.mSrc1) == RegType::Scl)
          src = ctx.lookup_val(si.mSrc1);
        else
          src = *find_mem(si.mSrc1);
        double* d = find_mem(si.mDst);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, 1, 1);
        MTXOP::sqrt_c_1d_mtxop_pd(d, src);
      }
      break;
      //TODO: expand operation here
      default: assert(false);
    }
  }
}

void memvaluateSSA(SSA& ssa, MemArena& arena){
  //GOTHERE
//  std::cout << "Before Optimization: " << std::endl;
//  std::cout << ssa;

  //TODO: optimize given SSA: dead code elimination etc
  local_value_numbering(ssa);

  //GOTHERE
//  std::cout << "After Local Value Numbering: " << std::endl;
//  std::cout << ssa;

  select_instruction(ssa);

  //GOTHERE
//  std::cout << "After Peephole Optimization: " << std::endl;
//  std::cout << ssa;

  RegSize regsize = estimate_register_size(ssa);
  std::vector<LiveSet> liveness = analyze_liveness(ssa);
  size_t minregs = estimate_cpu_local_registers(ssa, liveness);
  arena.reserve(regsize.rs * regsize.cs, minregs);
  MemInstrContext context(arena, regsize, minregs);

  //GOTHERE
//  std::cout << "temps: " << minregs << std::endl;

  std::vector<Instr> instr = local_register_allocation(ssa, context, liveness);

  //GOTHERE
  std::cout << "Final Instructions:" << std::endl;
  for (size_t i = 0; i < instr.size(); ++i)
    std::cout << instr[i];

  evaluate_cpu_instr(instr, context);
  release_ssa(context);
}

} //ML

#endif//ML_MEM_CODEGEN
