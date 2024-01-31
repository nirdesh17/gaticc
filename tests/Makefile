TEST_TARGETS=3x3-3x3 5x3-3x3 1x1-1x1 5x7-7x7 10x10-10x10 \
             conv_transform max_pooler average_pooler \
						 global_average_pooler gemm_transform \
             6x6-3x3-9x3 10x10-3x3-9x1 sasa_test \
						 ml_inference
LIB_PATH=../libsim.a
EXE_DIR=exe

CXX = g++
CXXFLAGS = -g -std=c++17 `pkg-config --cflags python3`
LDFLAGS = -lpython3.11 -Wl,--copy-dt-needed-entries -lprotoc -lprotobuf -lpthread

all: $(addsuffix .o,$(TEST_TARGETS))

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) $@ ${LIB_PATH} -o $* $(LDFLAGS)
	mv $* ${EXE_DIR}/

clean:
	rm -rf *.o ${EXE_DIR}/*
