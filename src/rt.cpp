#include "rt.h"
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
#include "onnx_parser.h"
#include "executor.h"
#include "ffi.h"
#include "tensor.h"
#include "instgen.h"

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
  if (!gbl_args.has_option("loadpy")) {
    log_fatal("Option --loadpy needs to be specified");
    gbl_args.print_usage();
  }

  if (!gbl_args.has_option("preprocfn")) {
    log_fatal("Option --preprocfn needs to be specified");
    gbl_args.print_usage();
  }

  if (!gbl_args.has_option("postprocfn")) {
    log_fatal("Option --postprocfn needs to be specified");
    gbl_args.print_usage();
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
  log_info("writing model weights to FPGA dram");
  //rah.write(data, size);
  log_info("write model weights complete");
  /* TODO: no way to know if it went through 
   * successfully to the fpga
   */
}

void Runner::infer_loop(Rah& rah, const Op::Parser& parser) {
  std::string mod_arg = gbl_args["loadpy"].as<std::string>();
  std::string mod_name = extract_basename(mod_arg).stem().string();
  std::filesystem::path mod_path = extract_dirname(mod_arg);
  std::cout << "starting engine\n";

  using inputT = float;
  using outputT = int8_t;
  PyEngine engine(mod_name, mod_path);
  log_info("reading input");
  for (int i = 0; i < 1; ++i) {
    CpuRunner cpu_runner;
    log_info("running preprocess on inputs");
    Tensor<outputT> *mid = cpu_runner.run<inputT,outputT>(engine, parser);
    log_info("preprocess finish");

    auto dims = mid->get_dims();
    uint32_t aligned_size = aligned_conv_input(dims);
    char *aligned_data = (char *) malloc((aligned_size + 1) * sizeof(char));
    std::cout << "aligned size " << aligned_size << '\n';
    BinBlob blob(aligned_data, aligned_size);
    blob.append_sa_input<outputT>(aligned_size, 0, mid);
    blob.write("pom.bin");
    log_info("start writing images to FPGA");
    //rah.write(aligned_data, aligned_size); 
    log_info("finish writing images to FPGA");
    std::exit(1);
  }
}

void Runner::run(Op::Parser &parser, const std::string& gml_file) {
  Rah rah;
  load_model(rah, gml_file);
  infer_loop(rah, parser);
}
