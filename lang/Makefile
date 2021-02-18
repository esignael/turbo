SRC_DIR = src
OBJ_DIR = build
INC_DIR = include

LIBXML2 = -lxml2 -I/usr/include/libxml2
LIBXCSP3 = -L../lib/XCSP3-CPP-Parser/lib -lxcsp3parser -I../lib/XCSP3-CPP-Parser/include
LIBS= $(LIBXML2) $(LIBXCSP3)

GCC=g++
GCC_FLAGS=-std=c++17 -I$(INC_DIR)

EXE = turboc

# Object files:
SOURCES = $(SRC_DIR)/turboc.cpp
INC_ONLY = $(INC_DIR)/*

## Compile ##

$(EXE): $(INC_ONLY) $(SOURCES)
	$(GCC) $(GCC_FLAGS) $(SOURCES) -o $@ $(LIBS)

# Clean objects in object directory.
clean:
	$(RM) $(EXE) $(EXE_SEQ)
