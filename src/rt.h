#pragma once

#include <cstddef>
#include <string>
#include "onnx_parser.h"

#define RAH_SO_STRING "librah.so"
#define RAH_APP_ID 1

/* Why re-invent streams?
 * stl streams do everything this does but extracting
 * a char* from them (so that it can be sent to librah)
 * leads to redundant copies. Fstream is a gentle cpp
 * wrapper around fread(), provides easy and cheap access
 * to the underlying char*.
 */
class Fstream {
  char *m_buf;
  size_t m_size;
public:
  Fstream(const std::string& filename);
  ~Fstream();
  const char *get_data() const;
  const size_t get_size() const;
};

class Rah {
  void *m_handle;

public:
  Rah();
  ~Rah();
  int write(const char *data, size_t size);
};

class Runner {
  void scan();
  void device_init();
  void load_model(Rah& rah, const std::string& gml_file);
  public:
    Runner();
    void run(Op::Parser &parser, const std::string& gml_file);
    void infer_loop(Rah& rah, const Op::Parser& parser);
};
