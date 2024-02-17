#pragma once

#include "onnx_parser.h"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <list>
#include <typeinfo>
#include <unistd.h>
/* from https://github.com/vietjtnguyen/argagg
 * for options parsing. See class Argparse for more info
 */
#include "argagg.h"

#define log_fatal(fmt, ...)                                                    \
  (log_fatal_func(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__))
#define log_info(fmt, ...)                                                     \
  (log_info_func(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__))

inline void log_fatal_func(const char *file, int line, const char *func,
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

/* Wrapper over argagg library */
class Argparse {
  const char *usage = "Usage: sysim [OPTIONS]\n";
  argagg::parser_results args;
  /* To extend, add a new definition here */
  argagg::parser argparser{
      /* name      invokation         description              expected
       *                                                       args */
      {{"help", {"-h", "--help"}, "get this help message nigga", 0},
       {"verbose", {"-v", "--v"}, "verbose", 0},
       {"onnx", {"--onnx"}, "load onnx file", 1},
       {"timeest",
        {"--timeest"},
        "print estimated time that a model would take based on FLOP counts "
        "(this does not account for latencies such as that of DRAM)",
        1},
       {"summary", {"--summary"}, "print a summary of the model", 0}}};

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

  void print_usage() const { std::cerr << usage << argparser; }

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

using Point = std::pair<int, int>;

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

template <typename T> void print_vec(const char *s, std::vector<T> const &v) {
  printf("%s: ", s);
  for (auto a : v) {
    std::cout << a << ' ';
  }
  std::cout << '\n';
}

template <typename T> void print_vec(const char *s, std::list<T> const &v) {
  printf("%s: ", s);
  for (auto const &a : v) {
    std::cout << a << ' ';
  }
  std::cout << '\n';
}

inline void print_vec_point(const char *s, std::vector<Point> const &v) {
  printf("%s: ", s);
  for (auto &p : v) {
    std::cout << p.first << ',' << p.second << ' ';
  }
  std::cout << '\n';
}

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

inline int sa_odims_row(Op::ConvParams const &cp) {
  // o = ((iw - kw + 2p) / s) + 1
  return ((cp.imap[0] - cp.k[0] + cp.pad[0] + cp.pad[2]) / cp.stride[0]) + 1;
}

inline int sa_odims_cols(Op::ConvParams const &cp) {
  return ((cp.imap[1] - cp.k[1] + cp.pad[1] + cp.pad[3]) / cp.stride[1]) + 1;
}

template <typename T = std::chrono::seconds>
class Timer {
  using Tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
  Tp m_start;
  Tp m_stop;

public:
  void start() { m_start = std::chrono::high_resolution_clock::now(); }
  void stop() { m_stop = std::chrono::high_resolution_clock::now(); }

  T difference() {
    return std::chrono::duration_cast<T>(m_stop - m_start);
  }
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

/* Parse a csv string made of integers of the form:
 * "9, 8, 8" and return a vector 
 */
inline std::vector<int> parse_csv_string(std::string &s) {
  std::vector<int> result;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    // Convert token to integer and add to result vector
    result.push_back(std::stoi(token));
  }
  return result;
}
