ROOT_DIR=.
LIBSIM_OBJ=${ROOT_DIR}/sim.o ${ROOT_DIR}/transformers.o 
OBJ=${LIBSIM_OBJ} ${ROOT_DIR}/main.o ${ROOT_DIR}/ffi.o ${ROOT_DIR}/ops.o
FLAGS=-g -std=c++14 `pkg-config --cflags python3`
TESTDIR=${ROOT_DIR}/tests
LDFLAGS=-lpython3.11

a: ${OBJ}
	g++ ${FLAGS} ${OBJ} -o a ${LDFLAGS}

main.o: ${ROOT_DIR}/main.cpp ${ROOT_DIR}/utils.h ${ROOT_DIR}/sim.h ${ROOT_DIR}/transformers.h
	g++ ${FLAGS} -c $<

sim.o: ${ROOT_DIR}/sim.cpp ${ROOT_DIR}/sim.h ${ROOT_DIR}/utils.h
	g++ ${FLAGS} -c $<

transformers.o: ${ROOT_DIR}/transformers.cpp ${ROOT_DIR}/transformers.h ${ROOT_DIR}/sim.h
	g++ ${FLAGS} -c $<

ffi.o: ${ROOT_DIR}/ffi.cpp ${ROOT_DIR}/ffi.h 
	g++ ${FLAGS} -c $<

ops.o: ${ROOT_DIR}/ops.cpp ${ROOT_DIR}/ops.h ${ROOT_DIR}/transformers.h
	g++ ${FLAGS} -c $<

test-compile: libsim
	make -C ${TESTDIR}

libsim: ${ROOT_DIR}/sim.cpp
	ar rcs ${ROOT_DIR}/libsim.a ${LIBSIM_OBJ}

test: ${LIBSIM_OBJ} test-compile
	${TESTDIR}/execute_tests.sh

clean:
	rm -rf *.o
