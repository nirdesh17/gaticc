#include "rt.h"
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

Fstream::Fstream(std::string& filename) {
  FILE *fp = fopen(filename.c_str(), "rb");
  check_c_return_val(fp, filename.c_str());
  struct stat sbuf;
  int err = stat(filename.c_str(), &sbuf);
  check_c_return_val(err, filename.c_str());
  m_size = sbuf.st_size;
  m_buf = (char *) malloc(sizeof(*m_buf) * m_size);
  check_c_return_val(m_buf, "malloc");
  size_t size_read = fread(m_buf, sizeof(*m_buf), m_size, fp);
  if (size_read != m_size) {
    log_fatal("couldn't read all %ld bytes, %ld bytes read", m_size, size_read);
  }
  fclose(fp);
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
