TEST_TARGETS=3x3-3x3 5x3-3x3 1x1-1x1 5x7-7x7 10x10-10x10 \
             conv_transform max_pooler average_pooler \
						 global_average_pooler gemm_transform \
             6x6-3x3-9x3 10x10-3x3-9x1 sasa_test \
						 ml_inference ffi_test
LIB_PATH=../libsim.a
EXE_DIR=exe

CXX = g++
CXXFLAGS = -g -std=c++17 `pkg-config --cflags python3`
PYTHONLIB = -lpython3.11
PROTOLIB = -Wl,--copy-dt-needed-entries -lprotoc -lprotobuf -lpthread

all: $(addsuffix .o,$(TEST_TARGETS))

sasa_test.o: sasa_test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

ml_inference.o: ml_inference.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB}
	mv $* ${EXE_DIR}/

ffi_test.o: ffi_test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB}
	mv $* ${EXE_DIR}/

float_sasa_test.o: float_sasa_test.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

float_sasa_onnx_parser.o: float_sasa_onnx_parser.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* ${PYTHONLIB} ${PROTOLIB}
	mv $* ${EXE_DIR}/

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* 
	mv $* ${EXE_DIR}/

clean:
	rm -rf *.o ${EXE_DIR}/*
