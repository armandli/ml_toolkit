app=test_linear2

SOURCES=test_linear2.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=
OTHER_OPT= -funroll-loops #not working well
OPT= -O3 -ftree-vectorize -march=native -mfpmath=sse

LIBS=-lopenblas -lpthread -lgfortran

INCLUDES=-I../ -I./ -I../../reader/

CXXFLAGS=-std=c++14 -MD -Wall -Wextra -pthread $(INCLUDES) $(LIBS) $(OPT) $(DEBUG)

COMPILER=g++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
