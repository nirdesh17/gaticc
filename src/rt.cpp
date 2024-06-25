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
  //m_handle = dlopen(RAH_SO_STRING, RTLD_LAZY);
  //check_c_return_val(m_handle, dlerror());
}

Rah::~Rah() {
  //dlclose(m_handle);
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

void Runner::check_args() {
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

void Runner::tensor_pool_init(const Op::Parser &parser) {
  int total_regs = parser.get_total_registers() + 1;
  tensor_pool.resize(total_regs);
  tensor_pool.free();
}

PyEngine Runner::create_pyengine() {
  log_info("starting PyEngine");
  std::string mod_arg = gbl_args["loadpy"].as<std::string>();
  std::string mod_name = extract_basename(mod_arg).stem().string();
  std::filesystem::path mod_path = extract_dirname(mod_arg);
  PyEngine engine(mod_name, mod_path);
  return engine;
}

std::string Runner::get_run_arg() {
  assert(gbl_args.has_option("run"));
  return gbl_args["run"].as<std::string>();
}

Runner::Runner(const Op::Parser &parser) {
  check_args();
  tensor_pool_init(parser);
  Rah rah;
  PyEngine engine = create_pyengine();
  std::string gml_file = get_run_arg();
  load_model(rah, gml_file);
  infer_loop(rah, engine, parser);
}

/* make sure correct bitstream is loaded & rah.service
 * is running
 */
void Runner::scan() {
  std::cout << "scanning for rah services no cap fr\n";
}

/* Loads aligned and padded weights to the FPGA's DRAM */
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


void Runner::infer_loop(Rah &rah, PyEngine &engine, const Op::Parser &parser) {
  using inputT = float;
  using outputT = int8_t;
  log_info("reading input");
  log_info("running preprocess on inputs");
  /* TODO: deduce the types dynamically */
  run<inputT, outputT, int8_t, float>(rah, engine, parser);
}

void Runner::fake_exec(Op::LayerBase *l) {
  if (tensor_pool.has_value(l->outputs.at(0))) {
    tensor_pool.free(l->outputs.at(0));
  }
}

DispatchTable Runner::get_dispatch_table() {
  return DispatchTable();
}
