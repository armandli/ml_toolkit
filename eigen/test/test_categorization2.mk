app=test_categorization2

SOURCES=test_categorization2.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=-g
OTHER_OPT= -funroll-loops #not working well
OPT= -O3 -ftree-vectorize -march=native -mfpmath=sse

LIBS=

INCLUDES=-I../ -I./ -I/usr/include/eigen3/

CXXFLAGS=-std=c++14 -MD -Wall -Wextra -pthread $(INCLUDES) $(LIBS) $(OPT) $(DEBUG)

COMPILER=g++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
