#include "rt.h"
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
#include "onnx_parser.h"

Fstream::Fstream(const std::string& filename) {
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

const char *Fstream::get_data() const {
  return m_buf;
}
const size_t Fstream::get_size() const {
  return m_size;
}

Rah::Rah() {
  m_handle = dlopen(RAH_SO_STRING, RTLD_LAZY);
  check_c_return_val(m_handle, dlerror());
}

Rah::~Rah() {
  dlclose(m_handle);
}

int Rah::write(const char *data, size_t size) {
  typedef int (*write_fn_t) (const uint8_t, const char*, const unsigned long);

  write_fn_t write_fn;
  write_fn = (write_fn_t) dlsym(m_handle, "rah_write");
  char *error = dlerror();
  if (error != NULL) {
    log_fatal("%s", error);
  }

  return (*write_fn)(RAH_APP_ID, data, size);
}

Runner::Runner() {
  if (!gbl_args.has_option("input_path")) {
    log_fatal("No input file provided");
  }
}

/* make sure correct bitstream is loaded & rah.service
 * is running
 */
void Runner::scan() {
  std::cout << "scanning for rah services no cap fr\n";
}

void Runner::load_model(Rah& rah, const std::string& gml_file) {
  scan();
  Fstream fp(gml_file);
  const char *data = fp.get_data();
  size_t size = fp.get_size();
  rah.write(data, size);
  /* TODO: no way to know if it went through */
}

void Runner::infer_loop(Rah& rah, const Op::Parser& parser) {
    
}

void Runner::run(Op::Parser &parser, const std::string& gml_file) {
  Rah rah;
  load_model(rah, gml_file);
  infer_loop(rah, parser);
}
