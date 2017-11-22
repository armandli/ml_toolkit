#ifndef ML_INSTR
#define ML_INSTR

#include <algorithm>
#include <ostream>

namespace ML {

struct RegName {
  static const size_t Len = 4;
  char name[Len];

  RegName(){
    memset(name, 0, Len);
  }
};
bool operator == (RegName a, RegName b){
  return strncmp(a.name, b.name, RegName::Len) == 0;
}
bool operator < (RegName a, RegName b){
  char as[RegName::Len], bs[RegName::Len];
  memset(as, 0, RegName::Len);
  memset(bs, 0, RegName::Len);
  memcpy(as, &a.name[1], RegName::Len - 1);
  memcpy(bs, &b.name[1], RegName::Len - 1);
  int ai = atoi(as), bi = atoi(bs);
  if (ai < bi) return true;
  else         return false;
}
std::ostream& operator << (std::ostream& out, const RegName& name){
  out << name.name[0] << name.name[1] << name.name[2] << name.name[3];
  return out;
}

struct RegNameHash {
  size_t operator()(RegName name) const noexcept {
    char copy[RegName::Len + 1];
    memset(copy, 0, RegName::Len + 1);
    memcpy(copy, name.name + 1, RegName::Len - 1);
    int hash = atoi(copy);
    return hash;
  }
};

enum class InstrType : unsigned {
  Add,
  Sub,
  EMul,
  EDiv,
  Dot,
  AddMC,
  SubMC,
  SubCM,
  EMulMC,
  EDivMC,
  EDivCM,
  GT,
  GTMC,
  GTCM,
  Mask,
  Trn,
  Not,
  Tanh,
  Softmax,
  Exp,
  Isnan,
  Isnan0,
  GT0MC,
  GT0CM,
  DRelu,
  //TODO: expand operation here
  //Following Instruction only applies to GPU side
  CopyTo,
  CopyFrom,
};

struct Instr {
  InstrType mType;
  RegName   mDst;
  RegName   mSrc1;
  RegName   mSrc2;

  Instr() = default;
  Instr(InstrType type, RegName dst, RegName src1, RegName src2):
    mType(type), mDst(dst), mSrc1(src1), mSrc2(src2) {}
};

std::ostream& operator << (std::ostream& out, const Instr& instr){
  out << instr.mDst << " <- ";
  switch (instr.mType){
    case InstrType::Trn:    out << "~"; break;
    case InstrType::Not:    out << "!"; break;
    case InstrType::Exp:    out << "exp("; break;
    case InstrType::Isnan:  out << "isnan("; break;
    case InstrType::Isnan0: out << "isnan0("; break;
    default:;
  }
  out << instr.mSrc1;
  switch (instr.mType){
    case InstrType::Add: case InstrType::AddMC:
      out << " + " << instr.mSrc2 << "\n";
    break;
    case InstrType::Sub: case InstrType::SubMC: case InstrType::SubCM:
      out << " - " << instr.mSrc2 << "\n";
    break;
    case InstrType::EMul: case InstrType::EMulMC:
      out << " * " << instr.mSrc2 << "\n";
    break;
    case InstrType::EDiv: case InstrType::EDivMC: case InstrType::EDivCM:
      out << " / " << instr.mSrc2 << "\n";
    break;
    case InstrType::GT: case InstrType::GTMC: case InstrType::GTCM:
      out << " > " << instr.mSrc2 << "\n";
    break;
    case InstrType::GT0MC: case InstrType::GT0CM:
      out << " >0 " << instr.mSrc2 << "\n";
    break;
    case InstrType::Mask:
      out << " mask " << instr.mSrc2 << "\n";
    break;
    case InstrType::Dot:
      out << " ^ " << instr.mSrc2 << "\n";
    break;
    case InstrType::Trn: case InstrType::Not:
      out << "\n";
    break;
    case InstrType::Exp: case InstrType::Isnan: case InstrType::Isnan0:
      out << ")\n";
    break;
    //TODO: expand operation here
    default:;
  }
  return out;
}

} //ML

#endif//ML_INSTR
