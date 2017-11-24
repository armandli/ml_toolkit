#ifndef ML_CUDA_CODEGEN
#define ML_CUDA_CODEGEN

#include <vector>

#include <cublas_v2.h>

#include <ml_cuda_common.h>
#include <ml_ssa_decl.h>
#include <ml_codegen.h>
#include <ml_cuda_arena.h>
#include <ml_cuda.cuh>

namespace ML {

size_t estimate_cuda_local_registers(SSA& ssa, const std::vector<LiveSet>& live){
  size_t res = 0;
  for (size_t i = 0; i < ssa.instructions.size(); ++i){
    size_t count = live[i].livein.size() + 1;
    res = std::max(res, count);
  }
  return res;
}

class CUDAInstrContext : public InstrContext {
  CUDArena& mArena;
  std::unordered_map<double*, RegName> mRegLookup;
  std::stack<RegName>                  mFree;
public:
  CUDAInstrContext(CUDArena& arena, RegSize rsz, size_t reg_count, SSAContext& sctx): InstrContext(), mArena(arena) {
    for (size_t i = 0; i < reg_count; ++i){
      RegName name = add_reg(RegData(arena.reg(i), rsz.rs, rsz.cs));
      mRegLookup.insert(std::make_pair(arena.reg(i), name));
    }
    std::unordered_map<const Memory*, RegName>& used = sctx.get_mtxmap();
    std::vector<RegName> free;
    for (size_t i = 0; i < reg_count; ++i){
      double* m = arena.reg(i);
      RegName name = mRegLookup(m);
      //TODO
      const Mtx* mtx = arena.get_cached_at(m);
      if (mtx == nullptr)
        free.push_back(name);
      else if (used.find(mtx) == used.end())
        mFree.push(name);
    }
    for (RegName r : free)
      mFree.push(r);
  }

  RegName get_free_reg(){
    assert(not mFree.empty());
    RegName ret = mFree.top();
    mFree.pop();

    RegData& dat = lookup_reg(ret);
    arena.free_cache(dat.mMem);
    return ret;
  }

  RegName get_reg_for(const Mtx& mtx){
    double* m = arena.get_cache(mtx);
    assert(m != nullptr);
    decltype(mRegLookup)::iterator it = mRegLookup.find(m);
    assert(it != mRegLookup.end());
    return (*it).second;
  }

  bool is_cached(const Mtx& mtx){
    return arena.get_cache(mtx) != nullptr;
  }

  void update_cache(RegName name, const Mtx& mtx){
    RegData& dat = lookup_reg(name);
    arena.register_cache(mtx, dat.mMem);
  }

  void uncache(RegName name){
    RegData& dat = lookup_reg(name);
    arena.free_cache(dat.mMem);
    mFree.push(name);
  }
};

//TODO: BUG!! you don't know the RegSize of the register that has been reused!!!
std::vector<Instr> cuda_register_allocation(SSA& ssa, CUDAInstrContext& ctx, const std::vector<LiveSet>& liveness){
  std::vector<Instr> ret;
  std::unordered_map<RegName, RegName, RegNameHash> tmap;

  class AllocateReg {
    std::unordered_map<RegName, RegName, RegNameHash>& tmap;
    CUDAInstrContext& ctx;
    std::vector<Instr>& dinstr;
  public:
    AllocateReg(std::unordered_map<RegName, RegName, RegNameHash>& tmap, CUDAInstrContext& ctx, std::vector<Instr>& dinstr):
      tmap(tmap), ctx(ctx), dinstr(dinstr) {}
    RegName operator()(RegName sname, SSAregData& data, bool is_dst){
      RegName ret;
      switch (sdat.mType){
        case SSAregType::Mtx: {
          std::unordered_map<RegName, RegName, RegNameHash>::iterator it = tmap.find(sname);
          if (it != tmap.end()){
            ret = (*it).second;
            break;
          }
          if (sdat.mMtxRef != nullptr){
            RegName mname = ctx.add_mem(*sdat.mMtxRef);
            if (ctx.is_cached(*sdat.mMtxRef))
              ret = ctx.get_reg_for(*sdat.mMtxRef);
            else {
              ret = ctx.get_free_reg();
              if (not is_dst)
                dinstr.emplace_back(Instr(InstrType::CopyTo, ret, mname, RegName()));
              ctx.update_cache(ret, *sdat.mMtxRef);
            }
          } else {
            ret = ctx.get_free_reg();
            tmap.insert(std::make_pair(sname, ret));
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
  } allocate_reg(tmap, ctx, ret);

  for (size_t i = 0; i < ssa.instructions.size(); ++i){
    Instr& si = ssa.instructions[i];
    const SSAregData& s1dat = ssa.context.lookup(si.mSrc1);
    RegName s1name = allocate_reg(si.mSrc1, s1dat, false);
    const SSAregData& s2dat = ssa.context.lookup(si.mSrc2);
    RegName s2name = allocate_reg(si.mSrc2, s2dat, false);
    const SSAregData& ddat  = ssa.context.lookup(si.mDst);
    RegName dname = allocate_reg(si.mDst, ddat, true);
    if (ctx.type(s1name) == RegType::Reg && liveness[i].liveout.find(si.mSrc1) == liveness[i].liveout.end() && s1dat.mMtxRef == nullptr){
      tmap.erase(si.mSrc1);
      ctx.uncache(s1name);
    }
    if (ctx.type(s2name) == RegType::Reg && liveness[i].liveout.find(si.mSrc2) == liveness[i].liveout.end() && s2dat.mMtxRef == nullptr){
      tmap.erase(si.mSrc2);
      ctx.uncache(s2name);
    }
    ret.emplace_back(Instr(si.mType, dname, s1name, s2name));
    //NOTE: we don't move data from GPU side to CPU side, the copying will only happen when it is explicitly requested by the matrix,
    //      the result will be kept at the GPU side in the temporary buffer, anything that's not copied before using the same arena
    //      for the next round of evluation could be lost
  }

  return ret;
}

//TODO: you'll need to use additional registers for temporary scratchpads for some cuda operations!!!
//TODO: figure out how to do copy between CPU and GPU asynchronously
void evaluate_cuda_instr(const std::vector<Instr>& instr, CUDAInstrContext& ctx){
  class MemFinder {
    CUDAInstrContext& ctx;
  public:
    MemFinder(CUDAInstrContext& ctx): ctx(ctx) {}
    double* operator()(RegName name){
      double* ret = nullptr;
      switch (ctx.type(name)){
        case RegType::Mem:
          ret = ComputeMtxCommunicator::get_data(*ctx.lookup_mem(name));
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
    CUDAInstrContext& ctx;
  public:
    SizeFinder(CUDAInstrContext& ctx): ctx(ctx) {}
    RegSize operator()(RegName name){
      RegSize ret = {0, 0};
      switch (ct.type(name)){
        case RegType::Mem: {
          const Mtx* m = ctx.lookup_mem(name);
          ret.rs = (*m).rows();
          ret.cs = (*m).cols();
        }
        break;
        case RegType::Reg: {
          RegData& dat = ctx.lookup_reg(name);
          ret.rs = data.mRows;
          ret.cs = data.mCols;
        }
        break;
        case RegType::Scl:
        case RegType::Nil:
        break;
        default: assert(false);
      }
    }
  } find_size(ctx);

  for (auto& si : instr){
    switch (si.mType){
      //TODO: add DRelu, DSigmoid, DTanh, Deriviative operation
      case InstrType::Add:
      case InstrType::Sub:
      case InstrType::EMul:
      case InstrType::EDiv:
      case InstrType::GT:
      case InstrType::Mask: {
        double* s1 = find_mem(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && s2 != nullptr && d != nullptr);
        RegSize s1size = find_size(si.mSrc1);
        assert(s1size.rs > 0 && s1size.cs > 0);
        ctx.set_reg_size(si.mDst, s1size.rs, s1size.cs);
        switch (si.mType){
          case InstrType::Add:  CUDA::add_1d_cuda_pd(d, s1, s2, roundup_row(s1size.rs), roundup_col(s1size.cs)); break;
          case InstrType::Sub:  CUDA::sub_1d_cuda_pd(d, s1, s2, roundup_row(s1size.rs), roundup_col(s1size.cs)); break;
          case InstrType::EMul: CUDA::emul_1d_cuda_pd(d, s1, s2, roundup_row(s1size.rs), roundup_col(s1size.cs)); break;
          case InstrType::EDiv: CUDA::ediv_2d_cuda_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_row(s1size.rs), roundup_col(s1size.cs)); break;
          case InstrType::GT:   CUDA::gt_2d_cuda_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_row(s1size.rs), roundup_col(s1size.cs)); break;
          case InstrType::Mask: CUDA::mask_2d_cuda_pd(d, s1, s2, s1size.rs, s1size.cs, roundup_row(s1size.rs), roundup_col(s1size.cs)); break; //TODO: is this right?
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
        ctx.set_reg_size(si.mDst, s1size.rs, s2size.cs);
        //TODO: reuse the cublas handle and all constant values
        int lda = s1size.rs, ldb = s1size.cs, ldc = s1size.rs;
        const double alpha = 1.;
        const double beta  = 0.;
        const double* palpha = &alpha;
        const double* pbeta  = &beta;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, s1size.rs, s2size.cs, s1size.cs, palpha, s1, lda, s2, ldb, pbeta, d, ldc);
        cublasDestroy(handle);
      }
      //TODO: add CELoss and CEAccuracy function
      break;
      case InstrType::SubMC:
      case InstrType::EDivMC:
      case InstrType::GTMC: {
        double* s1 = find_mem(si.mSrc1);
        double  s2 = ctx.lookup_val(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && d != nullptr);
        RegSize sz = find_size(si.mSrc1);
        assert(sz.rs > 0 && sz.cs > 0);
        ctx.set_reg_size(si.mDst, sz.rs, sz.cs);
        switch (si.mType){
          case InstrType::SubMC:  CUDA::sub_mc_2d_cuda_pd(d, s1, s2, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          case InstrType::EDivMC: CUDA::ediv_mc_2d_cuda_pd(d, s2, s2, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          case InstrType::GTMC:   CUDA::gt_mc_2d_cuda_pd(d, s1, s2, sz.rs, sz.cs, rondup_row(sz.rs), roundup_col(sz.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::SubCM:
      case InstrType::EDivCM:
      case InstrType::GTCM: {
        double  s1 = ctx.lookup_val(si.mSrc1);
        double* s2 = find_mem(si.mSrc2);
        double* d  = find_mem(si.mDst);
        assert(s2 != nullptr && d != nullptr);
        RegSize sz = find_size(si.mSrc2);
        assert(sz.rs > 0 && sz.cs > 0);
        ctx.set_reg_size(si.mDst, sz.rs, sz.cs);
        switch (si.mType){
          case InstrType::SubCM:  CUDA::sub_cm_2d_cuda_pd(d, s1, s2, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          case InstrType::EDivCM: CUDA::ediv_cm_2d_cuda_pd(d, s1, s2, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          case InstrType::GTCM:   CUDA::gt_cm_2d_cuda_pd(d, s1, s2, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::AddMC:
      case InstrType::EMulMC: {
        double val  = nan("");
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
        assert(sz.rs > 0 && sz.cs > 0);
        switch (si.mType){
          case InstrType::AddMC:  CUDA::add_const_2d_cuda_pd(d, arg, val, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          case InstrType::EMulMC: CUDA::emul_const_2d_cuda_pd(d, arg, val, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::L2Loss: {
        //TODO
      }
      break;
      case InstrType::Trn: {
        double* s1 = find_mem(si.mSrc1);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && d != nullptr);
        RegSize sz = find_size(s1);
        ctx.set_reg_size(si.mDst, sz.rs, sz.cs);
        CUDA::transpose_2d_cuda_pd(d, s1, roundup_row(sz.rs), roundup_col(sz.cs));
      }
      break;
      case InstrType::Tanh:
      case InstrType::Exp:
      case InstrType::Not:
      case InstrType::Isnan: {
        double* s1 = find_mem(si.mSrc1);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && d != nullptr);
        RegSize sz = find_size(s1);
        ctx.set_reg_size(si.mDst, sz.rs, sz.cs);
        switch (si.mType){
          case InstrType::Tanh:  CUDA::tanh_1d_cuda_pd(d, s1, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          case InstrType::Exp:   CUDA::exp_2d_cuda_pd(d, s1, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          case InstrType::Not:   CUDA::not_2d_cuda_pd(d, s1, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          case InstrType::Isnan: CUDA::isnan_2d_cuda_pd(d, s1, sz.rs, sz.cs, roundup_row(sz.rs), roundup_col(sz.cs)); break;
          default: assert(false);
        }
      }
      break;
      case InstrType::Softmax: {
        //TODO: how to deal with temporary buffer?
      }
      break;
      case InstrType::Sum: {
        //TODO: also need buffer, and final register has size 1 1
      }
      break;
      case InstrType::CopyTo:
      case InstrType::CopyFrom: {
        double* s1 = find_mem(si.mSrc1);
        double* d  = find_mem(si.mDst);
        assert(s1 != nullptr && d != nullptr);
        RegSize sz = find_size(s1);
        assert(sz.rs > 0 && sz.cs > 0);
        if (ctx.type(si.mDst) == RegType::Reg)
          ctx.set_reg_size(si.mDst, sz.rs, sz.cs);
        sz.rs = roundup_row(sz.rs);
        sz.cs = roundup_col(sz.cs);
        switch (si.mType){
          case InstrType::CopyTo:   CUDADBG(cudaMemcpy(d, s1, sz.rs * sz.cs * sizeof(double), cudaMemcpyHostToDevice)); break;
          case InstrType::CopyFrom: CUDADBG(cudaMemcpy(d, s1, sz.rs * sz.cs * sizeof(double), cudaMemcpyDeviceToHost)); break;
          default: assert(false);
        }
      }
      break;
      //TODO: expand operation here
      default: assert(false);
    }
  }
}

void cudavaluateSSA(SSA& ssa, CUDArena& arena){
  local_value_numbering(ssa);

  select_instruction(ssa);

  RegSize regsize = estimate_register_size(ssa);
  std::vector<LiveSet> lv = analyze_liveness(ssa);
  size_t minregs = estimate_cuda_local_registers(ssa, lv);
  arena.reset(regsize.rs * regsize.cs, minregs);
  CUDAInstrContext ctx(arena, regsize, minregs, ssa.context);

  std::vector<Instr> instr = cuda_register_allocation(ssa, ctx, lv);

  evaluate_cuda_instr(instr, ctx);
  release_ssa(ctx);
}

} //ML

#endif//ML_CUDA_CODEGEN
