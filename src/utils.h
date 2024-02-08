#pragma once

#include "onnx_parser.h"
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <list>
/* from https://github.com/vietjtnguyen/argagg
 * for options parsing. See class Argparse for more info
 */
#include "argagg.h"

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
};

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
  for (auto const &a : v) {
    // std::cout << a << ' ';
    printf("%d ", a);
  }
  printf("\n");
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

template <typename expectedT, typename computedT>
bool generate_report(const char *test_name, std::vector<expectedT> &expected,
                     std::vector<computedT> &computed) {
  printf("---------------------------------\n");
  printf("Test Name: %s\n", test_name);
  bool status = (expected == computed);
  printf("Status: %s\n", (status) ? "Pass" : "Fail");
  // print_vec("Expected: ", expected);
  // print_vec("Computed: ", computed);
  return status;
}

// void print_vec_point(const char *s, std::vector<Point> const &v);

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
       {"onnx", {"--onnx"}, "load onnx file", 1}}};

public:
  void parse(int argc, char *argv[]) {
    if (argc < 2) {
      std::cerr << usage << argparser;
      log_fatal("Too few arguments");
    }
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

inline int sa_odims_row(Op::ConvParams const &cp) {
  // o = ((iw - kw + 2p) / s) + 1
  return ((cp.imap[0] - cp.k[0] + cp.pad[0] + cp.pad[2]) / cp.stride[0]) + 1;
}

inline int sa_odims_cols(Op::ConvParams const &cp) {
  return ((cp.imap[1] - cp.k[1] + cp.pad[1] + cp.pad[3]) / cp.stride[1]) + 1;
}
