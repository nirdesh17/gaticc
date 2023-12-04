TEST_TARGETS=3x3-3x3.o 5x3-3x3.o 1x1-1x1.o 5x7-7x7.o 10x10-10x10.o \
			 conv_transform.o gemm_transform.o 10x10-3x3-9x1.o \
			 6x6-3x3-9x3.o
LIB_PATH=../libsim.a
EXE_DIR=exe

all: ${TEST_TARGETS} 

3x3-3x3.o: 3x3-3x3.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o 3x3-3x3
	@mv 3x3-3x3 ${EXE_DIR}/

5x3-3x3.o: 5x3-3x3.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o 5x3-3x3
	@mv 5x3-3x3 ${EXE_DIR}/

1x1-1x1.o: 1x1-1x1.cpp
	g++ -c $<  -o $@
	g++ $@ ${LIB_PATH} -o 1x1-1x1
	@mv 1x1-1x1 ${EXE_DIR}/

5x7-7x7.o: 5x7-7x7.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o 5x7-7x7
	@mv 5x7-7x7 ${EXE_DIR}/

10x10-10x10.o: 10x10-10x10.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o 10x10-10x10
	@mv 10x10-10x10 ${EXE_DIR}/

conv_transform.o: conv_transform.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o conv_transform
	@mv conv_transform ${EXE_DIR}/

gemm_transform.o: gemm_transform.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o gemm_transform
	@mv gemm_transform ${EXE_DIR}/

10x10-3x3-9x1.o: 10x10-3x3-9x1.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o 10x10-3x3-9x1
	@mv 10x10-3x3-9x1 ${EXE_DIR}/

6x6-3x3-9x3.o: 6x6-3x3-9x3.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o 6x6-3x3-9x3
	@mv 6x6-3x3-9x3 ${EXE_DIR}/
