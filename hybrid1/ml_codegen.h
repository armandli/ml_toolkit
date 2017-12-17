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
      case InstrType::Add:        pr[0] = 0; break;
      case InstrType::Sub:        pr[0] = 1; break;
      case InstrType::EMul:       pr[0] = 2; break;
      case InstrType::EDiv:       pr[0] = 3; break;
      case InstrType::Dot:        pr[0] = 4; break;
      case InstrType::AddMC:      pr[0] = 5; break;
      case InstrType::SubMC:      pr[0] = 6; break;
      case InstrType::SubCM:      pr[0] = 7; break;
      case InstrType::EMulMC:     pr[0] = 8; break;
      case InstrType::EDivMC:     pr[0] = 9; break;
      case InstrType::EDivCM:     pr[0] = 10; break;
      case InstrType::Trn:        pr[0] = 11; break;
      case InstrType::Not:        pr[0] = 12; break;
      case InstrType::Exp:        pr[0] = 13; break;
      case InstrType::Isnan:      pr[0] = 14; break;
      case InstrType::Mask:       pr[0] = 15; break;
      case InstrType::GT:         pr[0] = 16; break;
      case InstrType::GTMC:       pr[0] = 17; break;
      case InstrType::GTCM:       pr[0] = 18; break;
      case InstrType::Isnan0:     pr[0] = 19; break;
      case InstrType::GT0MC:      pr[0] = 20; break;
      case InstrType::GT0CM:      pr[0] = 21; break;
      case InstrType::DRelu:      pr[0] = 22; break;
      case InstrType::CELoss:     pr[0] = 23; break;
      case InstrType::CEAccuracy: pr[0] = 24; break;
      case InstrType::Sum:        pr[0] = 25; break;
      case InstrType::L2Loss:     pr[0] = 26; break;
      case InstrType::Sqrt:       pr[0] = 27; break;
      case InstrType::DSS:        pr[0] = 28; break;
      case InstrType::AddCC:      pr[0] = 29; break;
      case InstrType::SubCC:      pr[0] = 30; break;
      case InstrType::EMulCC:     pr[0] = 31; break;
      case InstrType::EDivCC:     pr[0] = 32; break;
      case InstrType::MSELoss:    pr[0] = 33; break;
      case InstrType::SqrtC:      pr[0] = 34; break;
      case InstrType::MSEAccuracy:pr[0] = 35; break;
      case InstrType::Abs:        pr[0] = 36; break;
      case InstrType::Trn1Dot:    pr[0] = 37; break;
      case InstrType::Trn2Dot:    pr[0] = 38; break;
      case InstrType::Trn3Dot:    pr[0] = 39; break;
      //TODO: expand operation here
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

    switch (recon.mType){
      // ascending register number order
      case InstrType::Add: case InstrType::EMul: case InstrType::AddCC: case InstrType::EMulCC:
        if (recon.mSrc2 < recon.mSrc1)
          std::swap(recon.mSrc1, recon.mSrc2);
      break;
      // constant value always goes second
      case InstrType::AddMC: case InstrType::EMulMC: case InstrType::L2Loss:
        if (ssa.context.lookup(instr.mSrc1).mType == SSAregType::Scl)
          std::swap(recon.mSrc1, recon.mSrc2);
      break;
      // cases where order does matter
      case InstrType::Sub: case InstrType::EDiv: case InstrType::Dot: case InstrType::GT: case InstrType::Mask:
      case InstrType::DRelu: case InstrType::CELoss: case InstrType::CEAccuracy: case InstrType::DSS:
      case InstrType::Trn1Dot: case InstrType::Trn2Dot: case InstrType::Trn3Dot:
      case InstrType::SubCC: case InstrType::EDivCC: case InstrType::MSELoss: case InstrType::MSEAccuracy:
      case InstrType::SubMC: case InstrType::SubCM: case InstrType::EDivMC: case InstrType::EDivCM:
      case InstrType::GTMC: case InstrType::GTCM: case InstrType::GT0MC: case InstrType::GT0CM:
      case InstrType::Trn: case InstrType::Not: case InstrType::Tanh: case InstrType::Softmax:
      case InstrType::Exp: case InstrType::Isnan: case InstrType::Isnan0: case InstrType::Sum:
      case InstrType::Sqrt: case InstrType::SqrtC: case InstrType::Abs:
        /* DO NOTHING */
      break;
      //TODO: expand operation here
      // cases where we should not see during local value numbering
      default: assert(false);
    }

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
      const SSAregData& s1Dat  = ssa.context.lookup(ssa.instructions[i].mSrc1);
      const SSAregData& s2Dat  = ssa.context.lookup(ssa.instructions[i].mSrc2);
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

//TODO: write an automatic tool for instruction selection based on pattern matching

//TODO: the instruction selector algorithm is severely flawed. there are many possible mathematical 
//  identities that can be discovered and transformed to make more optimal instructions. need to 
//  research into this.

// At this point we can only have hand coded pattern matcher
//benefit:
//  reduce both register pressure as well as computation speed by condensing well known operations 
//  into bigger CISC instructions that can do a faster job
void select_instruction(SSA& ssa){
  std::vector<Instr> nins, segment;
  std::vector<bool> done(ssa.instructions.size(), false);
  std::vector<LiveSet> lv = analyze_liveness(ssa);
  const size_t maxLookahead = 16;

  class InstructionMatcher {
    std::vector<Instr>&         segment;
    std::vector<bool>&          done;
    const SSA&                  ssa;
    const std::vector<LiveSet>& lv;
    size_t                      maxLookahead;

    size_t next_use(size_t idx){
      for (size_t i = idx + 1; i < std::min(ssa.instructions.size(), idx + maxLookahead); ++i)
        if (done[i] == false &&
            (ssa.instructions[i].mSrc1 == ssa.instructions[idx].mDst ||
             ssa.instructions[i].mSrc2 == ssa.instructions[idx].mDst))
          return i;
      return std::numeric_limits<size_t>::max();
    }

    size_t prev_def(size_t idx, size_t eidx, RegName reg, bool isDone){
      if (idx == 0) return std::numeric_limits<size_t>::max();
      for (size_t i = idx - 1; i > eidx; --i)
        if (done[i] == isDone && ssa.instructions[i].mDst == reg)
          return i;
      if (done[eidx] == isDone && ssa.instructions[eidx].mDst == reg)
        return eidx;
      return std::numeric_limits<size_t>::max();
    }

    void consumeSigmoid3(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EDivCM){
        segment.push_back(instr);
        done[idx] = true;
        const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
        if (s1dat.mVal != 1.)
          return;
        RegName src1;
        if (ssa.context.lookup(segment[0].mSrc1).mType == SSAregType::Mtx)
          src1 = segment[0].mSrc1;
        else
          src1 = segment[0].mSrc2;
        RegName src2 = ssa.context.gen();
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::Sigmoid, dst, src1, src2));
      }
    }

    void consumeSigmoid2(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::AddMC){
        segment.push_back(instr);
        done[idx] = true;
        const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
        const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
        if ( not ((s1dat.mType == SSAregType::Scl && s1dat.mVal == 1.) ||
                  (s2dat.mType == SSAregType::Scl && s2dat.mVal == 1.)))
          return;
        consumeSigmoid3(next_use(idx));
      }
    }

    void consumeSigmoid1(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::Exp){
        segment.push_back(instr);
        done[idx] = true;
        consumeSigmoid2(next_use(idx));
      }
    }

    void consumeDTanh2(size_t idx, RegName tmp){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EMul){
        done[idx] = true;
        RegName src2 = segment[0].mSrc1;
        RegName src1;
        if (instr.mSrc1 == tmp) src1 = instr.mSrc2;
        else                    src1 = instr.mSrc1;
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::DTanh, dst, src1, src2));
      }
    }

    bool consumeL2Loss3(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EMulCC){
        done[idx] = true;
        RegName src1 = segment[0].mSrc1;
        RegName src2;
        if (instr.mSrc1 == segment[1].mDst)
          src2 = instr.mSrc2;
        else
          src2 = instr.mSrc1;
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::L2Loss, dst, src1, src2));
        return true;
      }
      return false;
    }

    bool consumeL2Loss2(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EMulCC){
        const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
        const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
        if (not ((s1dat.mType == SSAregType::Scl && s1dat.mVal == 0.5) ||
                 (s2dat.mType == SSAregType::Scl && s2dat.mVal == 0.5)))
          return false;
        segment.push_back(instr);
        if (consumeL2Loss3(next_use(idx))){
          done[idx] = true;
          return true;
        }
        segment.pop_back();
        return false;
      }
      return false;
    }

    void consumeDTanh1L2Loss1(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::SubCM){
        segment.push_back(instr);
        done[idx] = true;
        const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
        if (s1dat.mVal != 1.)
          return;
        consumeDTanh2(next_use(idx), instr.mDst);
      } else if (instr.mType == InstrType::Sum){
        if (consumeL2Loss2(next_use(idx)))
          done[idx] = true;
      }
    }

    bool consumeDSigmoid4(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::SubCM){
        const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
        if (s1dat.mVal != 1.)
          return false;
        if (instr.mSrc2 != segment[0].mSrc1 && instr.mSrc2 != segment[0].mSrc2)
          return false;
        segment.push_back(instr);
        done[idx] = true;
        return true;
      }
      return false;
    }

    void consumeDSigmoid1(size_t idx, size_t pidx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EMul){
        Instr& pinstr = segment[0];
        size_t oidx;
        if (pinstr.mDst == instr.mSrc1)
          oidx = prev_def(idx, pidx, instr.mSrc2, false);
        else
          oidx = prev_def(idx, pidx, instr.mSrc1, false);
        if (oidx == std::numeric_limits<size_t>::max())
          return;
        if (consumeDSigmoid4(oidx)){
          done[idx] = true;
          RegName src2 = segment[1].mSrc2;
          RegName src1;
          if (pinstr.mSrc1 == src2)
            src1 = pinstr.mSrc2;
          else
            src1 = pinstr.mSrc1;
          RegName dst = instr.mDst;
          segment.clear();
          segment.emplace_back(Instr(InstrType::DSigmoid, dst, src1, src2));
        }
      }
    }

    bool consumeDSigmoid3(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EMul){
        if (instr.mSrc1 != segment[0].mSrc2 && instr.mSrc2 != segment[0].mSrc2)
          return false;
        segment.push_back(instr);
        done[idx] = true;
        return true;
      }
      return false;
    }

    void consumeDSigmoid2(size_t idx, size_t pidx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EMul){
        Instr& pinstr = segment[0];
        size_t oidx;
        if (pinstr.mDst == instr.mSrc1)
          oidx = prev_def(idx, pidx, instr.mSrc2, false);
        else
          oidx = prev_def(idx, pidx, instr.mSrc1, false);
        if (oidx == std::numeric_limits<size_t>::max())
          return;
        if (consumeDSigmoid3(oidx)){
          done[idx] = true;
          RegName src2 = segment[0].mSrc2;
          RegName src1;
          if (segment[1].mSrc1 == src2)
            src1 = segment[1].mSrc2;
          else
            src1 = segment[1].mSrc1;
          RegName dst = instr.mDst;
          segment.clear();
          segment.emplace_back(Instr(InstrType::DSigmoid, dst, src1, src2));
        }
      }
    }

    void consumeDeriviative2(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::Isnan0){
        done[idx] = true;
        RegName src1 = segment[0].mSrc1, src2 = segment[0].mSrc2;
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::Deriviative, dst, src1, src2));
      }
      //TODO: alternative way of evaluating isnan
    }

    //TODO: MSE Accuracy algebra is fairly complicated with many possible
    //  forms, we're just dealing with the form that I keep using to make
    //  this simple
    bool consumeMSEAccuracy4(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EMulCC){
        const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
        const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
        if (not ((s1dat.mType == SSAregType::Scl && s1dat.mVal == 0.5) ||
                 (s2dat.mType == SSAregType::Scl && s2dat.mVal == 0.5)))
          return false;
        done[idx] = true;
        RegName src1 = segment[0].mSrc1;
        RegName src2 = segment[0].mSrc2;
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::MSEAccuracy, dst, src1, src2));
        return true;
      }
      return false;
    }

    bool consumeMSEAccuracy3(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EDivCC){
        const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
        const SSAregData& p1dat = ssa.context.lookup(segment[0].mSrc1);
        if (s2dat.mType != SSAregType::Scl || p1dat.mRows != s2dat.mVal) return false;
        done[idx] = true;
        RegName src1 = segment[0].mSrc1;
        RegName src2 = segment[0].mSrc2;
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::MSEAccuracy, dst, src1, src2));
        return true;
      }
      return false;
    }

    bool consumeMSEAccuracy2(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EMulCC){
        const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
        const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
        if (not ((s1dat.mType == SSAregType::Scl && s1dat.mVal == 0.5) ||
                 (s2dat.mType == SSAregType::Scl && s2dat.mVal == 0.5)))
          return false;
        if (consumeMSEAccuracy3(next_use(idx))){
          done[idx] = true;
          return true;
        }
      } else if (instr.mType == InstrType::EDivCC){
        const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
        const SSAregData& p1dat = ssa.context.lookup(segment[0].mSrc1);
        if (s2dat.mType != SSAregType::Scl || p1dat.mRows != s2dat.mVal) return false;
        if (consumeMSEAccuracy4(next_use(idx))){
          done[idx] = true;
          return true;
        }
      }
      return false;
    }

    bool consumeMSELoss1MSEAccuracy1(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EDivCC){
        const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
        const SSAregData& p1dat = ssa.context.lookup(segment[0].mSrc1);
        if (s2dat.mType != SSAregType::Scl || p1dat.mRows != s2dat.mVal) return false;

        done[idx] = true;
        RegName src1 = segment[0].mSrc1;
        RegName src2 = segment[0].mSrc2;
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::MSELoss, dst, src1, src2));
        return true;
      } else if (instr.mType == InstrType::SqrtC){
        if (consumeMSEAccuracy2(next_use(idx))){
          done[idx] = true;
          return true;
        }
      }
      return false;
    }

    bool consumeDSS2(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return false;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::Sum){
        done[idx] = true;

        if (consumeMSELoss1MSEAccuracy1(next_use(idx))) return true;

        RegName src1 = segment[0].mSrc1;
        RegName src2 = segment[0].mSrc2;
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::DSS, dst, src1, src2));
        return true;
      }
      return false;
    }

    void consumeDeriviative1DSS1(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::EDivMC){
        segment.push_back(instr);
        done[idx] = true;
        Instr& pinstr = segment[0];
        const SSAregData& pdat = ssa.context.lookup(pinstr.mSrc1);
        const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
        if (pdat.mRows != s2dat.mVal)
          return;
        consumeDeriviative2(next_use(idx));
      }
      if (instr.mType == InstrType::EMul && instr.mSrc1 == instr.mSrc2){
        if (consumeDSS2(next_use(idx)))
          done[idx] = true;
      }
    }

    void consumeTrnDot1P1(size_t idx){
      if (idx == std::numeric_limits<size_t>::max()) return;
      const Instr& instr = ssa.instructions[idx];
      if (instr.mType == InstrType::Dot){
        //TODO: need to check the result of transpose is no longer being used
        //not generating any instr, generate in part 2
        segment.clear();
      }
    }

    void consumeTrnDot1P2(size_t idx){
      const Instr& instr = ssa.instructions[idx];
      size_t p1idx = prev_def(idx, 0, instr.mSrc1, true);
      size_t p2idx = prev_def(idx, 0, instr.mSrc2, true);
      const Instr& s1instr = ssa.instructions[p1idx];
      const Instr& s2instr = ssa.instructions[p2idx];
      if (s1instr.mType == InstrType::Trn && s2instr.mType == InstrType::Trn){
        RegName src1, src2;
        if (s1instr.mDst == instr.mSrc1){
          src1 = s1instr.mSrc1;
          src2 = s2instr.mSrc1;
        } else {
          src1 = s2instr.mSrc1;
          src2 = s1instr.mSrc1;
        }
        RegName dst = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::Trn3Dot, dst, src1, src2));
      } else if (s1instr.mType == InstrType::Trn){
        RegName src1 = s1instr.mSrc1;
        RegName src2 = instr.mSrc2;
        RegName dst  = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::Trn1Dot, dst, src1, src2));
      } else if (s2instr.mType == InstrType::Trn){
        RegName src1 = instr.mSrc1;
        RegName src2 = s2instr.mSrc1;
        RegName dst  = instr.mDst;
        segment.clear();
        segment.emplace_back(Instr(InstrType::Trn2Dot, dst, src1, src2));
      }
    }

    void state0(size_t idx){
      const Instr& instr = ssa.instructions[idx];
      segment.push_back(instr);
      done[idx] = true;
      switch (instr.mType){
        case InstrType::EMulMC: {
          const SSAregData& s1dat = ssa.context.lookup(instr.mSrc1);
          const SSAregData& s2dat = ssa.context.lookup(instr.mSrc2);
          if ((s1dat.mType == SSAregType::Scl && s1dat.mVal == -1.) ||
              (s2dat.mType == SSAregType::Scl && s2dat.mVal == -1.))
            consumeSigmoid1(next_use(idx));
        }
        break;
        case InstrType::EMul:
          if (instr.mSrc1 == instr.mSrc2)
            consumeDTanh1L2Loss1(next_use(idx));
          else
            consumeDSigmoid1(next_use(idx), idx);
        break;
        case InstrType::SubCM:
          consumeDSigmoid2(next_use(idx), idx);
        break;
        case InstrType::Sub:
          consumeDeriviative1DSS1(next_use(idx));
        break;
        case InstrType::Trn:
          consumeTrnDot1P1(next_use(idx));
        break;
        case InstrType::Dot:
          consumeTrnDot1P2(idx); //terminal instruction
        break;
        default:;
      }
    }
  public:
    InstructionMatcher(std::vector<Instr>& segment, std::vector<bool>& done, const SSA& ssa, const std::vector<LiveSet>& lv, size_t maxLookahead):
      segment(segment), done(done), ssa(ssa), lv(lv), maxLookahead(maxLookahead) {}
    void operator()(size_t idx){
      segment.clear();
      state0(idx);
    }
  } match_at(segment, done, ssa, lv, maxLookahead);
  
  for (size_t i = 0; i < ssa.instructions.size(); ++i){
    if (done[i]) continue;
    match_at(i);
    for (auto& instr : segment)
      nins.emplace_back(std::move(instr));
  }

  ssa.instructions = nins;
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
protected:
  std::unordered_map<RegName, const Memory*, RegNameHash> mMemMap;
  std::unordered_map<const Memory*, RegName>              mMtxMap;
  std::unordered_map<RegName, RegData, RegNameHash>       mRegMap;
  std::unordered_map<RegName, double, RegNameHash>        mConstMap;
  std::unordered_map<double, RegName>                     mValMap;
private:
  int                                                     mMemCount;
  int                                                     mRegCount;
  int                                                     mConstCount;

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

  void set_reg_size(RegName name, size_t rows, size_t cols){
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

  RegName add_reg(const RegData& data){
    RegName name = nextRegName();
    mRegMap.insert(std::make_pair(name, data));
    return name;
  }

  RegName add_mem(const Memory& mtx){
    decltype(mMtxMap)::iterator it = mMtxMap.find(&mtx);
    if (it == mMtxMap.end()){
      RegName name = nextMemName();
      mMemMap.insert(std::make_pair(name, &mtx));
      mMtxMap.insert(std::make_pair(&mtx, name));
      return name;
    } else
      return (*it).second;
  }

  RegName add_const(double val){
    decltype(mValMap)::iterator it = mValMap.find(val);
    if (it == mValMap.end()){
      RegName name = nextConstName();
      mConstMap.insert(std::make_pair(name, val));
      mValMap.insert(std::make_pair(val, name));
      return name;
    } else
      return (*it).second;
  }

  RegName add_nil(){
    RegName ret;
    ret.name[0] = 'n';
    return ret;
  }

  std::unordered_map<const Memory*, RegName>&  memMap(){
    return mMtxMap;
  }

  const Memory* lookup_mem(RegName name){
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
  static void clear_ssa(const Memory& mtx){
    mtx.clear_ssa();
  }
  static double* get_data(const Memory& mtx){
    return mtx.data();
  }
};

void release_ssa(InstrContext& ctx){
  std::unordered_map<const Memory*, RegName>& mtxes = ctx.memMap();
  for (std::unordered_map<const Memory*, RegName>::iterator it = mtxes.begin(); it != mtxes.end(); ++it){
    const Memory* pm = (*it).first;
    ComputeMtxCommunicator::clear_ssa(*pm);
  }
}

} //ML

#endif//ML_CODEGEN
