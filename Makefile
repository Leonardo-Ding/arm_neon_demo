INCLUDES :=
LIBRARIES :=
COMPILER ?= g++

all: build

build: neon_sgemm

neon_sgemm.o:neon_sgemm.cpp
	$(COMPILER) $(INCLUDES) -O3 -o $@ -c $<
neon_sgemm: neon_sgemm.o
	$(COMPILER) -o $@ $+ $(LIBRARIES)

clean:
	rm -f neon_sgemm.o neon_sgemm
