#include "pch.h"
#include "rt.h"
// #include <cstdlib>
// #include <stdio.h>
// #include <stdlib.h>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <unistd.h>
// #include <dlfcn.h>
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
size_t Fstream::get_size() const {
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
  write_fn_t write_fn = get_dlsym<write_fn_t>(m_handle, "rah_write");
  return (*write_fn)(RAH_APP_ID, data, size);
}

int Rah::read(char *data, size_t size) {
  std::cout << "rah read called \n";
  typedef int (*read_fn_t) (const uint8_t, const char*, const unsigned long);
  read_fn_t read_fn;
  read_fn = (read_fn_t) dlsym(m_handle, "rah_read");
  char *error = dlerror();
  if (error != NULL) {
    log_fatal("%s", error);
  }
  return (*read_fn)(RAH_APP_ID, data, size);
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

void Runner::tensor_pool_init() {
  int total_regs = m_parser->get_total_registers() + 1;
  tensor_pool.resize(total_regs);
  tensor_pool.free();
}

void Runner::pyengine_init() {
  log_info("starting PyEngine");
  std::string mod_arg = gbl_args["loadpy"].as<std::string>();
  std::string mod_name = extract_basename(mod_arg).stem().string();
  std::filesystem::path mod_path = extract_dirname(mod_arg);
  m_engine = new PyEngine(mod_name, mod_path);
}

std::string Runner::get_run_arg() {
  assert(gbl_args.has_option("run"));
  return gbl_args["run"].as<std::string>();
}

Runner::Runner(Op::Parser &parser): m_parser {&parser} {
  check_args();
  tensor_pool_init();
  pyengine_init();

  Rah rah;
  std::string gml_file = get_run_arg();
  Fstream fp(gml_file);
  load_model(rah, fp);
  infer_loop(rah, fp);
}

Runner::~Runner() {
  delete m_engine;
}

/* make sure correct bitstream is loaded & rah.service
 * is running
 * TODO: implement this, will probably require bitman?
 */
void Runner::scan() {
  std::cout << "scanning for rah services no cap fr\n";
}

/* Loads aligned and padded weights to the FPGA's DRAM */
void Runner::load_model(Rah& rah, const Fstream &fp) {
  scan();
  const char *data = fp.get_data();
  size_t size = fp.get_size();
  log_info("writing model weights to FPGA dram");
  rah.write(data, size);
  log_info("write model weights complete");
  /* TODO: no way to know if it went through 
   * successfully to the fpga
   */
}


void Runner::infer_loop(Rah &rah, const Fstream &fp) {
  log_info("Types are being hardcoded in inferloop");
  using inputT = float;
  using outputT = int8_t;
  log_info("reading input");
  log_info("running preprocess on inputs");
  HashedDispatchTable hdt(fp);
  /* TODO: deduce the types dynamically */
  run<inputT, outputT, int8_t, float>(rah, hdt);
}

void Runner::fake_exec(Op::LayerBase *l) {
  if (tensor_pool.has_value(l->outputs.at(0))) {
    tensor_pool.free(l->outputs.at(0));
  }
}

void Runner::read_uart(BinBlob &blob) {
  PyObject *args = Py_BuildValue("()");
  PyObject *ret = m_engine->call_func("read_uart", args);

  std::vector<uint8_t> rr = np2v<uint8_t>(ret);
  assert(rr.size() == blob.get_size());
  char *data = blob.get_data();
  for (int i = 0; i < rr.size(); ++i) {
    data[i] = rr.at(i);
  }
}

void Runner::receive_output(Rah &rah, Op::LayerBase *l) {
  int expected_hash = string_hash(l->name);
  auto expected_dims = l->aligned_output();

  uint32_t expected_data_size = prod(expected_dims.begin(), expected_dims.end(), 1) *
    Op::tpdt_sizeof(l->output_type);
  uint32_t expected_packet_size = io_tensor_packet_size(expected_data_size);

  log_info("expected packet size in receive output: %d", expected_packet_size);

  BinBlob blob(expected_packet_size);
  read_uart(blob);
  //rah.read(blob.get_data(), expected_packet_size);
  const unsigned char *data = (const unsigned char *) blob.get_data();
  
  for (int i = 0; i < expected_packet_size; ++i) {
    if (i % 16 == 0 && i != 0) {
      printf("\n");
    }
	  printf("0x%02x, ", data[i]);
  }

  check_dwp_header(data, expected_packet_size, expected_data_size, expected_hash);
  //check_dwp_footer(data, expected_packet_size, 0 /* expected data size */, 0 /* expected hash */);
  const unsigned char *real_data = data + DWP_HEADER_BYTES;
  if (l->output_type == onnx::TensorProto_DataType_INT8) {
    receive_output_aux<int8_t>(real_data, expected_dims, l);
  } else if (l->output_type == onnx::TensorProto_DataType_UINT8) {
    receive_output_aux<uint8_t>(real_data, expected_dims, l);
  } else {
    log_fatal("can't compute with tensor of type %s", Op::get_tensorproto_dtype_name(l->output_type));
  }
}


HashedDispatchTable::HashedDispatchTable(const Fstream &fp) {
  const unsigned char *data = (const unsigned char *)fp.get_data();
  size_t size = fp.get_size();
  assert(size > DWP_HEADER_BYTES);
  uint32_t dwp_header = extract_byte<uint32_t>(data, size, 0, 4);
  uint32_t ds = extract_byte<uint32_t>(data, size, 4, 8); /* in bytes */
  uint32_t addr = extract_byte<uint32_t>(data, size, 8, 12);
  assert(dwp_header == DWP_SOP);
  int total_instructions = (ds / (INST_SIZE_BITS / 8));
  /* i starts at 1 to skip the zeroth instruction */
  int inst_bytes = (INST_SIZE_BITS / 8);
  assert(size >= (DWP_HEADER_BYTES + (total_instructions * inst_bytes)));
  int ptr = DWP_HEADER_BYTES + inst_bytes;
  for (int i = 1; i < total_instructions; ++i) {
    std::bitset<INST_SIZE_BITS> inst =
        extract_bitset<INST_SIZE_BITS>(data, size, ptr,
                                                      ptr + inst_bytes);
    int opcode = extract_opcode(inst);
    if (opcode == OP_OutputBlock) {
      int dispatch_en = bitset_range_get<OutputBlock_DispatchEn_COUNT>(
          inst, OutputBlock_DispatchEn_LOW, OutputBlock_DispatchEn_HIGH);
      if (dispatch_en) {
        int dispatch_id = bitset_range_get<OutputBlock_DispatchID_COUNT>(
            inst, OutputBlock_DispatchID_LOW, OutputBlock_DispatchID_HIGH);
        tbl.push_back(dispatch_id);
      }
    }
    ptr = ptr + inst_bytes;
  }
}

bool HashedDispatchTable::should_dispatch(const Op::LayerBase *l) const {
  int hashed = string_hash(l->name); 
  auto itr = std::find(tbl.begin(), tbl.end(), hashed); 
  if (itr != tbl.end()) {
    return true;
  }
  return false;
}
