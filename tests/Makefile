TEST_TARGETS=3x3-3x3.o 5x3-3x3.o 1x1-1x1.o 5x7-7x7.o 10x10-10x10.o \
			 conv_transform.o max_pooler.o average_pooler.o global_average_pooler.o gemm_transform.o  \
			 6x6-3x3-9x3.o 10x10-3x3-9x1.o sasa_test.o
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

max_pooler.o: max_pooler.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o max_pooler -g
	@mv max_pooler ${EXE_DIR}/

average_pooler.o: average_pooler.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o average_pooler -g
	@mv average_pooler ${EXE_DIR}/

global_average_pooler.o: global_average_pooler.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o global_average_pooler -g
	@mv global_average_pooler ${EXE_DIR}/

conv_transform.o: conv_transform.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o conv_transform
	@mv conv_transform ${EXE_DIR}/

gemm_transform.o: gemm_transform.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o gemm_transform
	@mv gemm_transform ${EXE_DIR}/

sasa_test.o: sasa_test.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o sasa_test -g
	@mv sasa_test ${EXE_DIR}/

10x10-3x3-9x1.o: 10x10-3x3-9x1.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o 10x10-3x3-9x1
	@mv 10x10-3x3-9x1 ${EXE_DIR}/

6x6-3x3-9x3.o: 6x6-3x3-9x3.cpp
	g++ -c $< -o $@
	g++ $@ ${LIB_PATH} -o 6x6-3x3-9x3
	@mv 6x6-3x3-9x3 ${EXE_DIR}/
