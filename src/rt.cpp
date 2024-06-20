#include "rt.h"
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>

Fstream::Fstream(std::string& filename) {
  std::unique_ptr<FILE> fp {fopen(filename.c_str(), "r")};
  check_c_return_val(fp.get());
  struct stat sbuf;
  int err = stat(filename.c_str(), &sbuf);
  check_c_return_val(err);
  m_size = sbuf.st_size;
  m_buf = (char *) malloc(sizeof(*m_buf) * m_size);
  check_c_return_val(m_buf);
  size_t size_read = fread(m_buf, m_size, sizeof(*m_buf), fp.get());
  if (size_read != m_size) {
    log_fatal("couldn't read all %ld bytes", m_size);
  }
}

Fstream::~Fstream() {
  free(m_buf);
}

Runner::Runner() {
  if (!gbl_args.has_option("input_path")) {
    log_fatal("No input file provided");
  }
}

void Runner::run(Op::Parser &parser, const std::string& gml_file) {
  
}
