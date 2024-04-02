TEST_TARGETS=3x3-3x3 5x3-3x3 1x1-1x1 5x7-7x7 10x10-10x10 \
             conv_transform conv_testpad max_pooler average_pooler \
						 global_average_pooler gemm_transform \
             6x6-3x3-9x3 10x10-3x3-9x1 sasa_test \
						 ffi_test float_sasa_test float_sasa_onnx_parser \
						 tensor_flatten mnist_inference
LIB_PATH=../libsim.a
EXE_DIR=exe

CXX = g++
CXXFLAGS = -O3 -g -std=c++17 `pkg-config --cflags python3`
PYTHONLIB = -lpython3.11
PROTOLIB = -Wl,--copy-dt-needed-entries -lprotoc -lprotobuf -lpthread

all: $(addsuffix .o,$(TEST_TARGETS))

sasa_test.o: sasa_test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

tensor_flatten.o: tensor_flatten.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

transpose_test.o: transpose_test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

mnist_inference.o: mnist_inference.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

float_sasa_test.o: float_sasa_test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

float_sasa_onnx_parser.o: float_sasa_onnx_parser.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

sasa_ex.o: sasa_ex.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB}
	mv $* ${EXE_DIR}/

clean:
	rm -rf *.o ${EXE_DIR}/*
