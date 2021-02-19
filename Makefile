CXX=nvcc
CPPFLAGS=-I nvbio/ -I nvbio/contrib/cub/ -L nvbio/build/nvbio -lnvbio

nvbio-benchmarks: nvbio-benchmarks.cu
	$(CXX) $^ -o $@ $(CPPFLAGS) -O3
nvbio-benchmarks-dbg: nvbio-benchmarks.cu
	$(CXX) $^ -DDEBUG -o $@ $(CPPFLAGS) -ggdb

.ONESHELL:
nvbio-install:
	git submodule update --init --recursive
	cd nvbio
	git submodule update --init --recursive
	mkdir build && cd build
	CXX=g++-6 CC=gcc-6 cmake -DGPU_ARCHITECTURE=sm_70 ..
	make -j8

clean:
	rm -f nvbio-benchmarks nvbio-benchmarks-dbg
