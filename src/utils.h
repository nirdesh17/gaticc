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
       {"onnx", {"--onnx"}, "load onnx file", 1},
       {"timeest",
        {"--timeest"},
        "print estimated time that a model would take based on FLOP counts "
        "(this does not account for latencies such as that of DRAM)"
        "\n\tArgs: [comma separated arch config]"
        "\n\tEx: --timeest 9,8,8",
        1},
       {"sim",
        {"--sim"},
        "Simulate inference on an input. Use options like --onnx, --loadpy, --preprocfn, --postprocfn to load weights/inputs to the simulator",
        0},
       {"dump-output",
        {"--dump-output"},
        "Dump Outputs produced by the "
        "simulator. Args: [all | none | comma separated layer names]",
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
       {"summary", {"--summary"}, "print a summary of the model", 0}}};

    const char *usage_examples = "Examples:\n"
    "\tRun simulation over a model and inputs\n"
    "\t./sysim --onnx path/to/model.onnx --sim --loadpy path/to/script.py --preprocfn \"preproc_func\" --postprocfn \"postprocfunc\" --venv-path path/to/venv/site-packages\n";

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
    std::cout << *itr << '\t';
    newline_cnt++;
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
  /* TODO: add a destructor */
};

template <typename T> void TensorPool::set(int index, T data) {
  pool.at(index) = data;
}

template <typename T> T TensorPool::get(int index) {
  assert(pool.at(index).has_value() && "pool at index does not have a value");
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
