app=test_categorization

SOURCES=test_categorization.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=-g
OTHER_OPT= -funroll-loops #not working well
OPT= -O3 -ftree-vectorize -march=native -mfpmath=sse

LIBS=

INCLUDES=-I../ -I./ -I/usr/include/eigen3/

#restrict Eigen temporaries
#DEFINES=-DEIGEN_NO_MALLOC

#test for runtime no malloc
#DEFINES=-DEIGEN_RUNTIME_NO_MALLOC

DEFINES=

CXXFLAGS=-std=c++14 -MD -Wall -Wextra -pthread $(INCLUDES) $(LIBS) $(OPT) $(DEBUG) $(DEFINES)

COMPILER=g++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
