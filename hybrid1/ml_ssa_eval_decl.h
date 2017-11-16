#ifndef ML_SSA_EVAL_DECL
#define ML_SSA_EVAL_DECL

#include <memory>

namespace ML {

class SSA;
class MemArena;

void memvaluateSSA(std::shared_ptr<SSA>, MemArena&);

} //ML

#endif//ML_SSA_EVAL_DECL
