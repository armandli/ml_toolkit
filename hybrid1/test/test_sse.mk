app=test_sse

SOURCES=test_sse.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=
OTHER_OPT= -funroll-loops #not working well
OPT= -O3 -ftree-vectorize -march=native -mfpmath=sse

LIBS=-lgtest -lgtest_main -lopenblas -lpthread -lgfortran

INCLUDES=-I../ -I./ -I/user/include

CXXFLAGS=-std=c++14 -MD -Wall -Wextra -pthread $(INCLUDES) $(LIBS) $(OPT) $(DEBUG)

COMPILER=g++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
