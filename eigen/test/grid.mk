app=grid

SOURCES=grid.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=-g
OPT= -O3 -ftree-vectorize -march=native -mfpmath=sse

LIBS=

INCLUDES=-I../ -I./ -I/usr/include/eigen3/

CXXFLAGS= -std=c++1y -Wall -Wextra $(INCLUDES) $(LIBS) $(OPT) $(DEBUG)

COMPILER=g++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
