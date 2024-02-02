ROOT_DIR = .
SRC_DIR = $(ROOT_DIR)/src
OBJ_DIR = $(ROOT_DIR)/obj
TEST_DIR = $(ROOT_DIR)/tests

SRC_FILES = main.cpp sim.cpp sasa.cpp transformers.cpp ffi.cpp ops.cpp onnx_parser.cpp
OBJ_FILES = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES)) $(OBJ_DIR)/onnx.pb.o
LIBSIM_OBJ_FILES = $(filter-out $(OBJ_DIR)/main.o,$(OBJ_FILES))

CXX = g++
# TODO: figure out how to specify lib paths for numpy
CXXFLAGS = -g -std=c++17 `pkg-config --cflags python3`
LDFLAGS = -lpython3.11 -Wl,--copy-dt-needed-entries -lprotoc -lprotobuf -lpthread
LD_LIBRARY_PATH = /usr/local/lib

all: a

a: $(OBJ_FILES)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# main does not contain a main.h file (handled separately)
$(OBJ_DIR)/main.o: ${SRC_DIR}/main.cpp ${SRC_DIR}/utils.h ${SRC_DIR}/sim.h ${SRC_DIR}/transformers.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# protoc generated files have .cc extension (handle separately)
$(OBJ_DIR)/onnx.pb.o: ${SRC_DIR}/onnx.pb.cc ${SRC_DIR}/onnx.pb.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# all other files with a *.{cpp,h} pair fall in this
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h $(SRC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

libsim: $(LIBSIM_OBJ_FILES)
	ar rcs $(ROOT_DIR)/libsim.a $^

test: libsim
	make -C $(TEST_DIR)

clean:
	rm -rf $(OBJ_DIR)/*.o a

run-tests: 
	$(TEST_DIR)/execute_tests.sh
