ROOT_DIR = .
SRC_DIR = $(ROOT_DIR)/src
OBJ_DIR = $(ROOT_DIR)/obj
TEST_DIR = $(ROOT_DIR)/tests

SRC_FILES = main.cpp sim.cpp ffi.cpp onnx_parser.cpp utils.cpp executor.cpp options.cpp
OBJ_FILES = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES)) $(OBJ_DIR)/onnx.pb.o
LIBSIM_OBJ_FILES = $(filter-out $(OBJ_DIR)/main.o,$(OBJ_FILES))

PYTHON_VERSION=$(shell python3 -c 'import sys; vv = sys.version_info[:2]; sys.stdout.write(f"{vv[0]}.{vv[1]}")')

CXX = g++

# Determine the operating system
UNAME_S := $(shell uname -s)

CXXFLAGS =  -O3 -g -std=c++17 `pkg-config --cflags python3`

ifeq ($(UNAME_S),Darwin)
	# TODO: Add generatlised support for mac os where paths will be the same for all versions of protobuf
	CXXFLAGS += -I/opt/homebrew/Cellar/protobuf/25.3_1/include -I/opt/homebrew/Cellar/abseil/20240116.1/include
	CXXFLAGS += -I/opt/homebrew/opt/boost/include
	LDFLAGS +=  -L/opt/homebrew/opt/python@${PYTHON_VERSION}/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/lib 
	# NOTNEEDED? same as -L?: -Wl,-rpath,/opt/homebrew/opt/python@${PYTHON_VERSION}/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/lib 
	LDFLAGS +=  -Wl,-undefined,dynamic_lookup
	LDFLAGS +=  -L/opt/homebrew/Cellar/protobuf/25.3_1/lib 
else ifeq ($(UNAME_S),Linux)
	LDFLAGS  +=  -Wl,--copy-dt-needed-entries
else
	$(error "Unknown OS: ${UNAME_S}")
endif

LDFLAGS += -lpython${PYTHON_VERSION} -lpthread -lprotobuf -lprotoc
LD_LIBRARY_PATH = /usr/local/lib

all: a

a: $(OBJ_FILES)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# main.cpp has no main.h (handled separately)
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
