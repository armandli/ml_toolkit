PROGRAM_NAME := test_cuda

program_CXX_SRCS :=
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}

program_CU_SRCS := test_cuda.cu
program_CU_OBJS := ${program_CU_SRCS:.cu=.cuo}

program_INCLUDE_DIRS := . ../ /user/include 

LIBS=-lgtest -lgtest_main -lopenblas -lpthread -lcurand -lcublas

# Compiler flags
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -std=c++14 -O3 -g -MD -Wall -Wextra -pthread -pedantic -march=native -mfpmath=sse $(LIBS)

GEN_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\" #Target CC 3.5, for example
NVFLAGS :=-std=c++14 -O3 -rdc=true $(LIBS) #rdc=true needed for separable compilation
NVFLAGS += $(GEN_SM61)
NVFLAGS += $(foreach includedir,$(program_CU_INCLUDE_DIRS),-I$(includedir))

CUO_O_OBJECTS := ${program_CU_OBJS:.cuo=.cuo.o}

OBJECTS = $(program_CU_OBJS) $(program_CXX_OBJS)

.PHONY: all clean distclean

all: $(PROGRAM_NAME) 

debug: CXXFLAGS = -std=c++11 -g -O0 -march=native -mfpmath=sse -Wall -pedantic -DDEBUG -Wall -Wextra -pthread $(LIB)
debug: NVFLAGS = -std=c++11 -O0 $(GEN_SM61) -g -G
debug: NVFLAGS += $(foreach includedir,$(program_CU_INCLUDE_DIRS),-I$(includedir))
debug: $(PROGRAM_NAME)

test_cuda.cuo: test_cuda.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

# This is pretty ugly...details below
# The program depends on both C++ and CUDA Objects, but storing CUDA objects as .o files results in circular dependency
# warnings from Make. However, nvcc requires that object files for linking end in the .o extension, else it will throw
# an error saying that it doesn't know how to handle the file. Using a non .o rule for Make and then renaming the file 
# to have the .o extension for nvcc won't suffice because the program will depend on the non .o file but the files in
# the directory after compilation will have a .o suffix. Thus, when one goes to recompile the code all files will be
# recompiled instead of just the ones that have been updated. 
#
# The solution below solves these issues by silently converting the non .o suffix needed by make to the .o suffix 
# required by nvcc, calling nvcc, and then converting back to the non .o suffix for future, dependency-based 
# compilation.
$(PROGRAM_NAME): $(OBJECTS) 
	@ for cu_obj in $(program_CU_OBJS); \
	do				\
		mv $$cu_obj $$cu_obj.o; \
	done				#append a .o suffix for nvcc
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ $(program_CXX_OBJS) $(CUO_O_OBJECTS)
	@ for cu_obj in $(CUO_O_OBJECTS); 	\
	do					\
		mv $$cu_obj $${cu_obj%.*};	\
	done				#remove the .o for make

clean:
	@- $(RM) $(PROGRAM_NAME) $(OBJECTS) *~ 

distclean: clean
