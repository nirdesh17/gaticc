#pragma once

#include <cstdio>
#include <cstdarg>

#define log_fatal(fmt, ...) (log_fatal_func(__FILE__, __LINE__, __func__, fmt ,##__VA_ARGS__))

inline void log_fatal_func(const char *file, int line, const char *func, const char *fmt, ...) {
  fprintf(stderr, "%s:%d: %s: ", file, line, func);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}
 
template<typename T>
void print_vec_vec(const char *s, std::vector<std::vector<T>> const &v) {
    printf("%s:\n", s);
    for (auto i: v) {
        for (auto j: i) {
            std::cout << j << '\t';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

template<typename T>
void print_vec(const char *s, std::vector<T> const &v) {
    printf("%s: ", s);
    for (auto const &a: v) {
        std::cout << a << ' ';
    }
    std::cout << '\n';
}

template<typename T>
void print_vec(const char *s, std::list<T> const &v) {
    printf("%s: ", s);
    for (auto const &a: v) {
        std::cout << a << ' ';
    }
    std::cout << '\n';
}

template<typename T>
std::vector<T> flatten(std::vector<std::vector<T>> const &vec)
{
    std::vector<T> flattened;
    for (auto const &v: vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

template<typename T>
bool generate_report(const char *test_name, std::vector<T>& expected, std::vector<T>& computed) {
    printf("---------------------------------\n");
    printf("Test Name: %s\n", test_name);
    bool status = (expected == computed);
    printf("Status: %s\n", (status) ? "Pass" : "Fail");
    print_vec("Expected: ", expected);
    print_vec("Computed: ", computed);
    return status;
}

//void print_vec_point(const char *s, std::vector<Point> const &v);
