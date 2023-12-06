ROOT_DIR=.
SRC_DIR=${ROOT_DIR}/src
LIBSIM_OBJ=${ROOT_DIR}/sim.o ${ROOT_DIR}/transformers.o 
OBJ=${LIBSIM_OBJ} ${ROOT_DIR}/main.o ${ROOT_DIR}/ffi.o ${ROOT_DIR}/ops.o ${ROOT_DIR}/onnx_parser.o ${ROOT_DIR}/onnx.pb.o
FLAGS=-g -std=c++17 `pkg-config --cflags python3`
TESTDIR=${ROOT_DIR}/tests
LDFLAGS=-lpython3.11 -Wl,--copy-dt-needed-entries -lprotoc -lprotobuf -lpthread 

a: ${OBJ}
	g++ ${FLAGS} ${OBJ} -o a ${LDFLAGS}

main.o: ${SRC_DIR}/main.cpp ${SRC_DIR}/utils.h ${SRC_DIR}/sim.h ${SRC_DIR}/transformers.h
	g++ ${FLAGS} -c $<

sim.o: ${SRC_DIR}/sim.cpp ${SRC_DIR}/sim.h ${SRC_DIR}/utils.h
	g++ ${FLAGS} -c $<

transformers.o: ${SRC_DIR}/transformers.cpp ${SRC_DIR}/transformers.h ${SRC_DIR}/sim.h
	g++ ${FLAGS} -c $<

ffi.o: ${SRC_DIR}/ffi.cpp ${SRC_DIR}/ffi.h 
	g++ ${FLAGS} -c $<

ops.o: ${SRC_DIR}/ops.cpp ${SRC_DIR}/ops.h ${SRC_DIR}/transformers.h
	g++ ${FLAGS} -c $<

onnx: ${ROOT_DIR}/onnx.proto
	protoc --cpp_out=${SRC_DIR} $<

onnx.pb.o: ${SRC_DIR}/onnx.pb.cc ${SRC_DIR}/onnx.pb.h onnx
	g++ ${FLAGS} -c $<

onnx_parser.o: ${SRC_DIR}/onnx_parser.cpp ${SRC_DIR}/onnx_parser.h
	g++ ${FLAGS} -c $<

test-compile: libsim
	make -C ${TESTDIR}

libsim: ${SRC_DIR}/sim.cpp
	ar rcs ${ROOT_DIR}/libsim.a ${LIBSIM_OBJ}

test: ${LIBSIM_OBJ} test-compile
	${TESTDIR}/execute_tests.sh

clean:
	rm -rf *.o
