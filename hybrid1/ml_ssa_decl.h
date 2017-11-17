#ifndef ML_SSA_DECL
#define ML_SSA_DECL

#include <cassert>
#include <cstring>
#include <vector>
#include <unordered_map>

#include <ml_instr.h>

namespace ML {

class MemArena;
namespace CUDA {
class CUDArena;
}

enum class SSAregType : unsigned {
  Scl,
  Mtx,
  Nil,
};

class Mtx;

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

  void clear(){
    mMtxMap.clear();
    mConstMap.clear();
    mRegData.clear();
    mCounter = 0;
  }
};

//TODO: hide private members
struct SSA {
  std::vector<Instr> instructions;
  SSAcontext         context;

  void clear(){
    instructions.clear();
    context.clear();
  }

  bool empty(){
    return instructions.size() == 0;
  }
};

void memvaluateSSA(SSA&, MemArena&);

} //ML

#endif//ML_SSA_DECL
