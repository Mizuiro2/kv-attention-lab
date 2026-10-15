CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all test clean
all: build/test_pool build/simulate

build:
	mkdir -p build

build/test_pool: cpp/test_pool.cpp cpp/kv_pool.hpp | build
	$(CXX) $(CXXFLAGS) cpp/test_pool.cpp -o $@

build/simulate: cpp/simulate.cpp cpp/kv_pool.hpp | build
	$(CXX) $(CXXFLAGS) cpp/simulate.cpp -o $@

test: build/test_pool
	./build/test_pool

clean:
	rm -rf build
