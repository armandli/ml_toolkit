#ifndef ML_SSA_DECL
#define ML_SSA_DECL

#include <memory>

namespace ML {

class SSA;
class MemArena;

void memvaluateSSA(std::shared_ptr<SSA>, MemArena&);

} //ML

#endif//ML_SSA_DECL
