ROOT_DIR = .
SRC_DIR = $(ROOT_DIR)/src
OBJ_DIR = $(ROOT_DIR)/obj
TEST_DIR = $(ROOT_DIR)/tests
DEBUG = 1

SRC_FILES = main.cpp sim.cpp ffi.cpp onnx_parser.cpp utils.cpp executor.cpp \
						options.cpp tensor.cpp instgen.cpp rt.cpp
PCH_SOURCES = sim.h ffi.h onnx.pb.h utils.h executor.h instgen.h options.h \
							onnx_parser.h rt.h
OBJ_FILES = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES)) $(OBJ_DIR)/onnx.pb.o
PCH_FILES = $(patsubst %.h,$(SRC_DIR)/%.h.gch,$(PCH_SOURCES))
LIBSIM_OBJ_FILES = $(filter-out $(OBJ_DIR)/main.o,$(OBJ_FILES))


PYTHON_VERSION=$(shell python3 -c 'import sys; vv = sys.version_info[:2]; sys.stdout.write(f"{vv[0]}.{vv[1]}")')

CXX = g++

# Determine the operating system
UNAME_S := $(shell uname -s)

NUMPY_INSTALL_PATH = /usr/lib/python${PYTHON_VERSION}/site-packages/numpy/
CXXFLAGS = -O3 -std=c++17 `pkg-config --cflags python3` -I${NUMPY_INSTALL_PATH}/_core/include -Wno-narrowing -DRAH_ENABLE=${RAH_ENABLE}

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g
endif

ifeq ($(UNAME_S),Darwin)
	PROTOBUF_PATH_MAC = `brew info protobuf | grep -m 1 'Cellar' | cut -d " " -f 1`
	ABSEIL_PATH_MAC = `brew info abseil | grep -m 1 'Cellar' | cut -d " " -f 1`
	BOOST_PATH_MAC = `brew info boost | grep -m 1 'Cellar' | cut -d " " -f 1`
	PYTHON_PATH_MAC = `brew info python | grep -m 1 'Cellar' | cut -d " " -f 1`

ONEONE = $(shell echo "${PYTHON_PATH_MAC}")

CXXFLAGS += -I$(PROTOBUF_PATH_MAC)/include -I$(ABSEIL_PATH_MAC)/include
CXXFLAGS += -I$(BOOST_PATH_MAC)/include
LDFLAGS +=  -L$(PYTHON_PATH_MAC)/Frameworks/Python.framework/Versions/Current/lib 
# NOTNEEDED? same as -L?: 
# -Wl,-rpath,/opt/homebrew/opt/python@${PYTHON_VERSION}/Frameworks/Python.framework/Versions/${PYTHON_VERSION}/lib 
LDFLAGS +=  -Wl,-undefined,dynamic_lookup
LDFLAGS +=  -L${PROTOBUF_PATH_MAC}/lib 
else ifeq ($(UNAME_S),Linux)
	LDFLAGS  +=  -Wl,--copy-dt-needed-entries
else
	$(error "Unknown OS: ${UNAME_S}")
endif

LDFLAGS += -lpython${PYTHON_VERSION} -lpthread -lprotobuf 
LD_LIBRARY_PATH = /usr/local/lib

all: a

a: $(PCH_FILES) $(OBJ_FILES)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) $(CXX) $(CXXFLAGS) $(OBJ_FILES) -o $@ $(LDFLAGS)

$(SRC_DIR)/%.h.gch: $(SRC_DIR)/%.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

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
	./execute_tests.sh
