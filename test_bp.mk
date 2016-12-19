app=test_bp

SOURCES=test_bp.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(app)

DEBUG=
OPT= -static -O3 -lm

LIBS=

INCLUDES=-I../ -I./

CXXFLAGS= -std=c++1y -Wall -Wextra $(INCLUDES) $(LIBS) $(OPT) $(DEBUG)

COMPILER=g++

$(app): %: %.o $(OBJECTS)
	$(COMPILER) $(CXXFLAGS) $^ -o $@

-include $(SOURCES:=.d) $(apps:=.d)

clean:
	-rm $(app) *.o *.d 2> /dev/null
