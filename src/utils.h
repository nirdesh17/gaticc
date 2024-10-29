#pragma once

#include <any>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <typeinfo>
#include <unistd.h>
#include <filesystem>
#include <map>
#include <cassert>
#include <bitset>
#include <variant>
#include <cstring>
#include <numeric>
#include <cerrno>
/* from https://github.com/vietjtnguyen/argagg
 * for options parsing. See class Argparse for more info
 */
#include "argagg.h"

#define log_fatal(fmt, ...)                                                    \
  (log_fatal_func(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__))
#define log_info(fmt, ...)                                                     \
  (log_info_func(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__))

[[noreturn]] inline void log_fatal_func(const char *file, int line, const char *func,
                           const char *fmt, ...) {
  fprintf(stderr, "%s:%d: %s: FATAL: ", file, line, func);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

inline void log_info_func(const char *file, int line, const char *func,
                          const char *fmt, ...) {
  fprintf(stderr, "%s:%d: %s: INFO: ", file, line, func);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
}

inline void check_c_return_val(int val, const char *err) {
  if (val != 0) {
    log_fatal("%s: %s", err, strerror(errno));
  }
}

inline void check_c_return_val(void* val, const char *err) {
  if (val == NULL) {
    log_fatal("%s: %s", err, strerror(errno));
  }
}

struct SaDims {
  int rows;
  int cols;
  int num;
};

/* Wrapper over argagg library */
class Argparse {
  const char *usage = "Usage: sysim [OPTIONS]\n";
  argagg::parser_results args;
  /* To extend, add a new definition here */
  argagg::parser argparser{
      /* name      invokation         description              expected
       *                                                       args */
      {{"help", {"-h", "--help"}, "print help and exit", 0},
       {"verbose", {"-v", "--v"}, "verbose", 0},
       {"timeest",
        {"--timeest"},
        "print estimated time that a model would take based on FLOP counts "
        "(this does not account for latencies such as that of DRAM)"
        "\n\tArgs: [comma separated arch config]"
        "\n\tEx: --timeest 9,8,8",
        1},
       {"info", {"-i", "--info"}, "Query information from model. Args: <onnx_model>", 1},
       {"sim",
        {"-s", "--sim"},
        "Simulate inference on an input. Use options like --onnx, --loadpy, --preprocfn, --postprocfn to load weights/inputs to the simulator",
        1},
       {"venv-path",
        {"--venv-path"},
        "Append venv-path to sys.path while loading the interpreter. Args: [ : "
        "separated path list]",
        1},
       {"loadpy",
        {"--loadpy"},
        "Load the python script mentioned in arg. Usually the script that'll contain pre/post process functions for --sim"
        "\n\tArgs: [script_name.py]",
        1},
       {"preprocfn",
        {"--preprocfn"},
        "Function that'll be called to get inputs that should be fed to the inference engine. Accepts no arguments, Returns a numpy array of atleast two dims, first being the batch and rest inputs i.e. (batch_size, ...)"
        "\n\tArgs: [func_name]",
        1},
       {"postprocfn",
        {"--postprocfn"},
        "Results from the inference engine would be handed to this function. Should expect (batch_size, ...) dimensional array"
        "\n\tArgs: [func_name]",
        1},
       {"input_path", {"--inputpath"}, "specify input to model as a resident file path (one file at a time)", 1},
       {"instgen", {"--instgen"}, "generate and print config instructions for an onnx model", 0},
       {"sa_arch", {"--sa-arch"}, "systolic array architecture. Args: [comma sep values]. accepts args like --timeest", 1},
       {"ramsize", {"--ramsize"}, "ram size in MB. Args: int. For ex, --ramsize 512", 1},
       {"vasize", {"--vasize"}, "Vector Array size. Usually equivalent to per cycle DRAM bandwidth in bytes. Args: int. For ex, --vasize 32", 1},
       {"pretty-print-blob", {"--pretty-print-blob"}, "pretty print entire blob", 0},
       {"pretty-print-inst", {"--pretty-print-inst"}, "pretty print only instructions", 0},
       {"pretty-print-inst-raw", {"--pretty-print-inst-raw"}, "pretty print instructions in raw hex", 0},
       {"output", {"--output", "-o"}, "write output to file. Args: filename. For ex, -o model.gml", 1},
       {"run", {"-r", "--run"}, "run inference on model. Args: <gml_file>.", 1},
       {"compile", {"-c", "--compile"}, "Compile onnx model into gml file. Args: <onnx_model>", 1},
       {"run_onnx", {"--run-onnx"}, "onnx model thorough which model.gml was generated. TODO: remove this", 1},
       {"dispatch", {"--dispatch"}, "comma separated list of layers for which outputs are required. Args: [all | none | comma separated layer names]", 1},
       {"dispatch_fn", {"--dispatch-fn"}, "python function that'll be passed tensors returned by dipatchable nodes", 1},
       {"receive-over-uart", {"--receive-over-uart"}, "receive over /dev/ttyUSB0. Args: baudrate (as set by the FPGA)", 1},
       {"compare-layer", {"--compare-layer"}, "compare layer with a golden value. The golden tensor can be generated by the simulator. Args: [dir where golden .npy tensor can be found]", 1},
       {"summary", {"--summary"}, "print a summary of the model", 0}}};

    const char *usage_examples = "Examples:\n"
    "\tRun simulation over a model and inputs\n"
    "\t./sysim -s model.onnx --inputpath image.jpg --loadpy src/ml_inference.py --preprocfn \"<pre_proc_fn>\""
      " --postprocfn \"<post_proc_fn>\" --venv-path ~/path/to/lib/python{version}/site-packages/ -v\n\n"
    "\tCreate a GML model file from onnx\n"
    "\t./sysim -c model.onnx -o model.gml --ramsize 512 --sa-arch 9,4,4 --vasize 32\n\n"
    "\tRun inference on FPGA\n"
    "\t./sysim -r model.gml --run-onnx model.onnx --inputpath img.jpg --loadpy src/ml_inference.py --preprocfn \"preprocess\" --postprocfn \"post_imagenet\" --venv-path ~/path/to/lib/python{version}/site-packages/ --sa-arch 9,4,4 --ramsize 512 --vasize 32\n\n"
    "\tPretty Print Generated Instructions\n"
    "\t./sysim -c model.onnx --ramsize 512 --sa-arch 9,4,4 --vasize 32 --pretty-print-inst\n\n"
    "\tPretty Print GML file\n"
    "\t./sysim -c model.onnx --ramsize 512 --sa-arch 9,4,4 --vasize 32 --pretty-print-blob\n\n"
    "\tPrint a summary of onnx file\n"
    "\t./sysim -i model.onnx --summary\n\n"
    "\tGet layer wise inference time estimates\n"
    "\t./sysim -i model.onnx --timeest 9,4,4\n\n";


public:
  void parse(int argc, char *argv[]) {
#if 0
    if (argc < 2) {
      std::cerr << usage << argparser;
      log_fatal("Too few arguments");
    }
#endif
    try {
      args = argparser.parse(argc, argv);
    } catch (const std::exception &e) {
      std::cerr << usage << argparser;
      log_fatal("%s", e.what());
    }
  }

  argagg::option_results &operator[](const std::string &name) {
    return args[name];
  }

  bool has_option(const std::string &name) const {
    return args.has_option(name);
  }

  void print_usage() const { std::cerr << usage << argparser << usage_examples; }
};

/* This is globally available for all functions. Alternatively,
 * an Argparse object could have been passed to each and every
 * contructor but this is the way I've decided to do it
 *
 * The Argparse::parse method is called on this object by main()
 * which in turn calls the underlying argagg functions.
 *
 * Functions looking to use args can simply call the subscript
 * operator[] on gbl_args.
 */
extern Argparse gbl_args;

struct Point {
  int first;
  int second;
  Point(int a, int b);
};
std::ostream& operator<<(std::ostream &os, const Point& p);

template <typename T> class Mat {
  std::vector<std::vector<T>> data;

public:
  std::vector<T> &at(int index) { return data.at(index); }
  const std::vector<T> &at(int index) const { return data.at(index); }
  T &at(int i, int j) { return data.at(i).at(j); }
  const T &at(int i, int j) const { return data.at(i).at(j); }
  void push_back(std::vector<T> &v) { data.push_back(v); }
  void push_back(std::vector<T> &&v) { data.push_back(v); }

  int size(int index) { return data.at(index).size(); };
  int size() const { return data.size(); };

  Mat(int size, const std::vector<T> &value) { data.resize(size, value); }
  Mat(int size) { data.resize(size); }
  Mat() {}

  std::vector<T> flatten() {
    std::vector<T> flattened;
    for (auto const &v : data) {
      flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
  }

  typename std::vector<std::vector<T>>::iterator begin() {
    return data.begin();
  }

  typename std::vector<std::vector<T>>::iterator end() { return data.end(); }

  void print() {
    for (auto i : data) {
      for (auto j : i) {
        std::cout << j << '\t';
      }
      std::cout << '\n';
    }
    std::cout << '\n';
  }
};

/* convert v into 2d array (Mat) of dims (rows,column) */
template <typename T> Mat<T> v2mat(std::vector<T> &v, int rows, int columns) {
  Mat<T> m;
  for (int i = 0; i < rows; ++i) {
    std::vector<T> vv;
    for (int j = 0; j < columns; ++j) {
      vv.push_back(v.at(i * columns + j));
    }
    m.push_back(vv);
  }
  return m;
}

template <typename T>
std::vector<T> mat2v(Mat<T> const &m, int rows, int columns) {
  std::vector<T> v;
  for (int i = 0; i < m.size(); i++) {
    for (int j = 0; j < m.at(0).size(); j++) {
      v.push_back(m.at(i, j));
    }
  }
  return v;
}

template <typename T>
void print_vec_vec(const char *s, std::vector<std::vector<T>> const &v) {
  printf("%s:\n", s);
  for (auto i : v) {
    for (auto j : i) {
      std::cout << j << '\t';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

/* TODO: use type_traits here */
/* Check if v belongs to the signed int family */
template <typename T> inline bool is_int_like(T v) {
  return typeid(v) == typeid(int) || typeid(v) == typeid(int8_t) ||
         typeid(v) == typeid(int16_t) || typeid(v) == typeid(int64_t) ||
         typeid(v) == typeid(long) || typeid(v) == typeid(long long);
}

template <typename T> inline bool is_unsigned_int_like(T v) {
  return typeid(v) == typeid(uint32_t) || typeid(v) == typeid(uint8_t) ||
         typeid(v) == typeid(uint16_t) || typeid(v) == typeid(uint64_t) ||
         typeid(v) == typeid(unsigned long) ||
         typeid(v) == typeid(unsigned long long);
}

template <typename T> inline bool is_float_like(T v) {
  return typeid(v) == typeid(float) || typeid(v) == typeid(double);
}

/* any container that overloads std::begin and std::end and operator<< on 
 * its elements should be printable. the name has been kept for legacy
 * reasons, makes sense to use on linear containers.
 */
template <typename Container> void print_vec(const char *s, Container const &v) {
  printf("%s: ", s);
  int newline_cnt = 0;
  std::cout << std::setprecision(8) << std::fixed;
  for (auto itr = std::begin(v); itr != std::end(v); ++itr) {
    /* print only 16 number on a single line */
    if (newline_cnt >= 9) {
      std::cout << '\n';
      newline_cnt = 0;
    }
    if (is_int_like<decltype(*itr)>(*itr) || is_unsigned_int_like<decltype(*itr)>(*itr)) {
      std::cout << (int) *itr << '\t';
    } else {
      std::cout << *itr << '\t';
    }
    newline_cnt++;
  }
  std::cout << '\n';
}


/* custom compare function to handle floats separately */
template <typename T> bool xcmp(T a, T b) {
  if (is_float_like(a)) {
    /* epsilon value suggests inquality of two floats is
     * fine uptill 3 digits precision */
    return (std::fabs(a - b) < 0.0005f);
  } else {
    return a == b;
  }
}

/* TODO: two types do not make sense */
template <typename expectedT, typename computedT>
bool generate_report(const char *test_name, std::vector<expectedT> &expected,
                     std::vector<computedT> &computed) {
#if 0
  for (int i = 0; i < expected.size(); ++i) {
      std::cout << i << ' ' << expected.at(i) << ' ' << computed.at(i) << '\n';
  }
#endif
  printf("---------------------------------\n");
  printf("Test Name: %s\n", test_name);
  bool status = false;
  assert(expected.size() == computed.size() && "expected - computed unequal");
#if 1
  for (int i = 0; i < expected.size(); ++i) {
    status = xcmp<expectedT>(expected.at(i), computed.at(i));
    if (!status) {
      std::cout << "Failing at " << i << " for " << expected.at(i) << ' '
                << computed.at(i) << '\n';
    }
  }
#endif
  printf("Status: %s\n", (status) ? "Pass" : "Fail");
  return status;
}

// void print_vec_point(const char *s, std::vector<Point> const &v);

template <typename T = std::chrono::seconds> class Timer {
  using Tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
  Tp m_start;
  Tp m_stop;

public:
  void start() { m_start = std::chrono::high_resolution_clock::now(); }
  void stop() { m_stop = std::chrono::high_resolution_clock::now(); }

  T difference() { return std::chrono::duration_cast<T>(m_stop - m_start); }
  void report(std::string msg) {
    std::cout << msg << difference().count() << '\n';
  }
  // TODO: reset function
};

class MemProf {
  double m_start;
  double m_stop;
  double m_vm;

public:
  void process_mem_usage(double &vm_usage, double &resident_set) {
    vm_usage = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
      std::string ignore;
      std::ifstream ifs("/proc/self/stat", std::ios_base::in);
      ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
          ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
          ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
          ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                        1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
  }

  void start() { this->process_mem_usage(m_vm, m_start); }

  void stop() { this->process_mem_usage(m_vm, m_stop); }

  /* Difference in KB */
  long difference() { return m_stop - m_start; }

  void report() {
    std::cout << "RSS: " << this->difference() << " KB, VM: " << m_vm
              << " KB\n";
  }
};

/* Parse a csv string of the form "9,8,8" and return a vector
 */
template <typename T> inline std::vector<T> parse_csv_string(std::string &s) {
  std::vector<T> result;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    result.push_back(token);
  }
  return result;
}

template <> inline std::vector<int> parse_csv_string(std::string &s) {
  std::vector<int> result;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    result.push_back(std::stoi(token));
  }
  return result;
}

class TensorPool {
  std::vector<std::any> pool;

public:
  template <typename T> void set(int index, T data);
  template <typename T> T get(int index);
  void free();
  void free(int index);
  bool has_value(int index);
  void resize(int size);
  void print() const;
  /* TODO: add a destructor */
};

template <typename T> void TensorPool::set(int index, T data) {
  pool.at(index) = data;
}

template <typename T> T TensorPool::get(int index) {
  assert(pool.at(index).has_value() && "pool at index does not have a value");
  if (pool.at(index).type() != typeid(T)) {
  	log_fatal("at index %d, expected type %s but got %s", index,
		pool.at(index).type().name(), typeid(T).name());
  }
  return std::any_cast<T>(pool.at(index));
}

/* like std::accumulate but calculates products 
 * TODO: use this in tensor.h
 */
template<class InputIt, class T>
T prod(InputIt first, InputIt last, T init) {
  T product = init;
  for (InputIt i = first; i != last; ++i) {
    product *= *i;
  }
  return product;
}

/* Add v1 and v2 and store into v1 */
template <typename T>
void add_vec(std::vector<T>& v1, const std::vector<T>& v2) {
  assert(v1.size() == v2.size());
  std::vector<T> ret(v1.size());
  for (int i = 0; i < v1.size(); ++i) {
    v1[i] = v1[i] + v2[i];
  }
}

/* path: such as "/usr/bin/file.txt"
 * returns: "file.txt"
 */
std::filesystem::path extract_basename(const std::string &path);
/* path: such as "/usr/bin/file.txt"
 * returns: "/usr/bin"
 */
std::filesystem::path extract_dirname(const std::string &path);

/* Container Concatenate */
template <typename Container>
Container concat(const Container &v1, const Container &v2) {
  Container ret;
  ret.insert(ret.begin(), v1.begin(), v1.end());
  ret.insert(ret.end(), v2.begin(), v2.end());
  return ret;
}

/* Element-wise multiplication */
template <typename T>
std::vector<T> operator*(const std::vector<T> &v1, const std::vector<T> &v2) {
  assert(v1.size() == v2.size());
  std::vector<T> ret(v1.size());
  for (int i = 0; i < v1.size(); ++i) {
    ret[i] = v1[i] * v2[i];
  }
  return ret;
}

/* strides arrays are used pre-dominantly in elementwise vector-to-vector
 * style multiplications and addition, thus makes sense to use a valarray 
 * here
 */
template <typename Container>
inline Container get_stride_from_shape(const Container &shape) {
  Container ret(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    ret[i] = prod(std::begin(shape)+i+1, std::end(shape), 1);
  }
  return ret;
}

template <typename Container>
inline Container get_stride_from_shape(const Container &&shape) {
  return get_stride_from_shape(shape);
}

bool is_broadcastable(const std::vector<int> &shape1, const std::vector<int>& shape2);

std::vector<int> get_dims_after_pad(std::vector<int> current_dims, const std::vector<int>& pad);

/* return true if i,j lie in pad section of a 2d segment */
bool islying(int i, int j, int rows, int cols, const std::vector<int> &pad);

template <typename variantT, typename vectorT>
std::vector<vectorT> variant2vec(const std::vector<variantT> &var) {
  std::vector<vectorT> ret;
  for (variantT i : var) {
    if (std::holds_alternative<uint8_t>(i)) {
      ret.push_back((vectorT) std::get<uint8_t>(i));
    } else if (std::holds_alternative<int8_t>(i)) {
      ret.push_back((vectorT) std::get<int8_t>(i));
    } else {
      log_fatal("cant deduce type for zero point");
    }
  }
  return ret; 
}

template <typename T>
std::vector<T> broadcast_vec(const std::vector<T> &in, int new_size) {
  if (in.size() == 1) {
    std::vector<T> ret(new_size);
    for (int i = 0; i < new_size; ++i) {
      ret[i] = in[0];
    }
    return ret;
  } else {
    assert(in.size() == new_size);
    return in;
  }
}

/* example:
 *  bitset_range_set(dest, src, 0, 3)
 * will set the first 4 least significant bits of dest by copying first four lsb
 * from src
 */
template <std::size_t b1N, std::size_t b2N>
void bitset_range_set(std::bitset<b1N>& dest, const std::bitset<b2N>& src, int start, int stop) {
  assert(stop - start + 1 == src.size());
  for (int i = 0; i < src.size(); ++i) {
    dest[start] = src[i];
    start++;
  }
}

template <std::size_t b1N, std::size_t b2N>
unsigned long bitset_range_get(const std::bitset<b2N>& src, int start, int stop) {
  std::bitset<b1N> ret;
  for (int i = start, j = 0; i < stop + 1; ++i, ++j) {
    ret[j] = src[i];
  }
  return ret.to_ulong();
}

template <typename T>
void assert_all_equal(const T *arr, int size) {
  assert(size > 0);
  T a = arr[0];
  for (int i = 0; i < size; ++i) {
    assert(arr[i] == a);
  }
}

int cmp_dims(const std::vector<int>& dim1, const std::vector<int>& dim2);

template <typename T>
inline T ceil_mod(T i, int m) {
  return (T) (std::ceil((float)i/(float)m) * m);
}

int count_digits(int a);
void print_table(const std::map<std::string, int>& tbl);

std::vector<int> get_sa_arch();
int get_va_size();

/* extract bytes from n to m and return them */
template <typename T, typename FromT>
T extract_byte(const FromT *data, size_t size, int n, int m) {
  assert(n>=0);
  assert(m>0);
  assert(n <= m);
  assert(m-n <= size);
  assert(m-n == sizeof(T));
  T ret = 0;
  for (int i = n, j = (sizeof(T)-1); i < m; ++i, --j) {
    T tmp = static_cast<T>(data[i]);
    tmp <<= (j*8);
    ret |= tmp;
  }
  return ret;
}

inline int string_hash(const std::string& s) {
  return std::accumulate(s.begin(), s.end(), 0);
}

template <size_t sz, typename T>
std::bitset<sz> extract_bitset(const T *data, size_t size, int n, int m) {
  assert(m - n == (sz/8));
  assert(m - n < size);
  std::bitset<sz> ret {0};
  for (int i = n, j = ((sz/8)-1); i < m; ++i, --j) {
    std::bitset<sz> tmp {data[i]};
    tmp <<= (j*8);
    ret |= tmp;
  }
  return ret;
}

