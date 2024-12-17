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
    log_fatal("couldn't read all {} bytes, {} bytes read\n", m_size, size_read);
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

/* convert a 32 bit integer into a 48 bit byte stream */
std::vector<char> cvt_32248(int v) {
  std::vector<char> buf;
  buf.push_back(static_cast<char>(0x00));
  buf.push_back(static_cast<char>(0x00));
  buf.push_back(static_cast<char>((v & 0xFF000000) >> 24));
  buf.push_back(static_cast<char>((v & 0x00FF0000) >> 16));
  buf.push_back(static_cast<char>((v & 0x0000FF00) >> 8));
  buf.push_back(static_cast<char>((v & 0x000000FF)));
  return buf;
}

static const std::vector<char> META_SOP = {0xff, 0xff, 0xff, 0xff, 0xff, 0x0ff};
static const std::vector<char> META_TYPE_RESET = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static const std::vector<char> META_TYPE_DISPATCH = {0x00, 0x00, 0x00, 0x00, 0x00, 0x01};
static const std::vector<char> META_TYPE_PAYLOAD_SIZE = {0x00, 0x00, 0x00, 0x00, 0x00, 0x02};

static const std::vector<char> META_CONST_DISPATCH_RAH = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static const std::vector<char> META_CONST_DISPATCH_UART = {0x00, 0x00, 0x00, 0x00, 0x00, 0x01};


RealRah::RealRah() {
  m_handle = dlopen(RAH_SO_STRING, RTLD_LAZY);
  if (m_handle == NULL) {
    log_fatal("dlopen(): {}: could not open {}, check if you've installed rah. \n"
              "Additionally, check "
              "if vaaman-fpga communication overlay has been configured "
              "properly (see "
              "https://docs.vicharak.in/vicharak_sbcs/vaaman/vaaman-linux/"
              "linux-configuration-guide/vicharak-config-tool/) ", dlerror(), RAH_SO_STRING);
  }
}

RealRah::~RealRah() {
  dlclose(m_handle);
}

int RealRah::write(const char *data, size_t size) {
  /* clear buffers before writing */
  typedef int (*clear_fn_t) (const uint8_t);
  clear_fn_t clear_fn = get_dlsym<clear_fn_t>(m_handle, "rah_clear_buffer");
  log_info("clear buffers before read\n");
  (*clear_fn)(RAH_APP_ID);

  typedef int (*write_fn_t) (const uint8_t, const char*, const unsigned long);
  write_fn_t write_fn = get_dlsym<write_fn_t>(m_handle, "rah_write");

  log_info("writing meta app, size {}\n", size);
  std::vector<char> size_buf = cvt_32248(size);
  write_meta(META_TYPE_PAYLOAD_SIZE, size_buf);

  log_info("writing via rah, size {}\n", size);
  /* send the actual data */
  int r = (*write_fn)(RAH_APP_ID, data, size);
  return r;
}

/* 
 * Lowest level MetaApp write.
 * TODO: document the META protocol
 * 'size' here is the size of payload in bytes 
 */
int RealRah::write_meta(const std::vector<char> &type,
                    const std::vector<char> &data) {

  std::vector<char> size_buf{cvt_32248(static_cast<int>(data.size()))};
  std::vector<char> packet;
  packet.insert(packet.end(), META_SOP.begin(), META_SOP.end());
  packet.insert(packet.end(), size_buf.begin(), size_buf.end());
  packet.insert(packet.end(), type.begin(), type.end());

  for (char i : data) {
    packet.push_back(i);
  }

  typedef int (*clear_fn_t) (const uint8_t);
  clear_fn_t clear_fn = get_dlsym<clear_fn_t>(m_handle, "rah_clear_buffer");
  (*clear_fn)(META_APP_ID);

  typedef int (*write_fn_t) (const uint8_t, const char*, const unsigned long);
  write_fn_t write_fn = get_dlsym<write_fn_t>(m_handle, "rah_write");
  int r = (*write_fn)(META_APP_ID, packet.data(), packet.size());
  return r;
}

int RealRah::read(char *data, size_t size) {

  typedef int (*read_fn_t) (const uint8_t, const char*, const unsigned long);
  read_fn_t read_fn;
  read_fn = (read_fn_t) dlsym(m_handle, "rah_read");
  char *error = dlerror();
  if (error != NULL) {
    log_fatal("{}\n", error);
  }
  log_info("reading via rah, size {}\n", size);
  return (*read_fn)(RAH_APP_ID, data, size);
}

int FakeRah::write_meta(const std::vector<char> &type,
                    const std::vector<char> &data) {
  return static_cast<int>(data.size());
}

int FakeRah::write(const char *data, size_t size) {
  return size;
}

int FakeRah::read(char *data, size_t size) {
  int m_ptr = 0;
  auto append_int = [&](uint32_t a) {
    /* reverse iteration for big endian */
    for (int i = sizeof(uint32_t) - 1; i >= 0; --i) {
      char c = get_byte(a, i);
      data[m_ptr++] = c;
    }
  };
  memset(data, 0, size);
  append_int(DWP_SOP);
  append_int((uint32_t)(size - (DWP_HEADER_BYTES * 2)));
  append_int((uint32_t)2108);

  int8_t c = 1;
  for (int i = m_ptr; i < size; ++i) {
    data[i] = c;
    c++;
  }
  return size;
}

void Runner::check_args() {
  if (!gbl_args.has_option("loadpy")) {
    log_fatal("Option --loadpy needs to be specified\n");
    gbl_args.print_usage();
  }

  if (!gbl_args.has_option("preprocfn")) {
    log_fatal("Option --preprocfn needs to be specified\n");
    gbl_args.print_usage();
  }

  if (!gbl_args.has_option("postprocfn")) {
    log_fatal("Option --postprocfn needs to be specified\n");
    gbl_args.print_usage();
  }
}

void Runner::tensor_pool_init() {
  int total_regs = m_parser->get_total_registers() + 1;
  tensor_pool.resize(total_regs);
  tensor_pool.free();
}

void Runner::pyengine_init() {
  log_info("starting PyEngine\n");
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

  std::string gml_file = get_run_arg();
  Fstream fp(gml_file);

  if (gbl_args.has_option("dry-run")) {
    FakeRah fr;
    load_model(fr, fp);
    infer_loop(fr, fp);
  } else {
    RealRah rr;
    load_model(rr, fp);
    infer_loop(rr, fp);
  }
}

Runner::~Runner() {
  delete m_engine;
}

/* make sure correct bitstream is loaded & rah.service
 * is running
 * TODO: implement this, will probably require bitman?
 */
void Runner::scan(Rah &rah) {
  log_info("scanning for rah services no cap fr\n");
  log_info("resetting FPGA\n");
  std::vector<char> empty;
  rah.write_meta(META_TYPE_RESET, empty);
}

/* Loads aligned and padded weights to the FPGA's DRAM */
void Runner::load_model(Rah& rah, const Fstream &fp) {
  scan(rah);
  const char *data = fp.get_data();
  size_t size = fp.get_size();

  log_info("setting dispatch type\n");
  if (gbl_args.has_option("receive-over-uart")) {
    rah.write_meta(META_TYPE_DISPATCH, META_CONST_DISPATCH_UART);
  } else {
    rah.write_meta(META_TYPE_DISPATCH, META_CONST_DISPATCH_RAH);
  }

  log_info("writing model weights to FPGA dram\n");
  rah.write(data, size);
  log_info("write model weights complete\n");
  /* TODO: no way to know if it went through 
   * successfully to the fpga
   */
}


void Runner::infer_loop(Rah &rah, const Fstream &fp) {
  log_warn("Types are being hardcoded in inferloop\n");
  using inputT = float;
  using outputT = int8_t;
  log_info("reading input\n");
  log_info("running preprocess on inputs\n");
  HashedDispatchTable hdt(fp);
  /* TODO: deduce the types dynamically */
  run<inputT, outputT, int8_t, float>(rah, hdt);
}

void Runner::fake_exec(Op::LayerBase *l) {
  if (tensor_pool.has_value(l->outputs.at(0))) {
    tensor_pool.free(l->outputs.at(0));
  }
}

void Runner::read_uart(BinBlob &blob, int uart_baud, int expected_size) {
  PyObject *args = Py_BuildValue("(ii)", uart_baud, expected_size);
  PyObject *ret = m_engine->call_func("read_uart", args);
  if (ret == NULL) {
  	log_fatal("read_uart failed don't know why\n");
  }
  Tensor<int8_t> *rr = np2t<int8_t>(ret);
  assert(rr->size() == blob.size());
  char *data = blob.get_data();
  for (int i = 0; i < rr->size(); ++i) {
    data[i] = rr->at(i);
  }
  delete rr;
  Py_XDECREF(ret);
  Py_XDECREF(args);
}

void Runner::receive_output(Rah &rah, Op::LayerBase *l) {
  int expected_hash = string_hash(l->name);
  uint32_t expected_data_size = 0;

  if (strcmp(l->op_type(), "QLinearConv") == 0) {
	std::vector<int> dims;
	Op::Layer::QLinearConv *cc = dynamic_cast<Op::Layer::QLinearConv *>(l);
	if (cc->pipelined_output_dims.size() != 0) {
		dims = cc->pipelined_output_dims;
	} else {
		dims = l->output_dims;
	}
	expected_data_size = aligned_conv_output(dims) * Op::tpdt_sizeof(l->output_type);
  } else if (strcmp(l->op_type(), "QLinearMatMul") == 0 || strcmp(l->op_type(), "QGemm") == 0) {
    expected_data_size = aligned_fc_io(l->output_dims) * Op::tpdt_sizeof(l->output_type);
  } else {
  	log_fatal("Unhandled layer of type: {}\n", l->op_type());
  }
  auto expected_dims = l->aligned_output();
  uint32_t expected_packet_size = io_tensor_packet_size(expected_data_size);

  log_info("expected packet size in receive output: {}\n", expected_packet_size);
  log_info("expected data size in receive output: {}\n", expected_data_size);

  BinBlob blob(expected_packet_size);

  if (gbl_args.has_option("receive-over-uart")) {
    int baud_rate = gbl_args["receive-over-uart"].as<int>();
    read_uart(blob, baud_rate, expected_packet_size);
  } else {
    rah.read(blob.get_data(), expected_packet_size);
  }

  const unsigned char *data = (const unsigned char *) blob.get_data();

  if (!gbl_args.has_option("dry-run")) {
    /* dry-run is a false traversal of the run loop used for debugging,
     * correctness is not really needed all that much
     */
    check_dwp_header(data, expected_packet_size, expected_data_size, expected_hash);
  }

  //check_dwp_footer(data, expected_packet_size, 0 /* expected data size */, 0 /* expected hash */);
  if (l->output_type == onnx::TensorProto_DataType_INT8) {
    const int8_t *real_data = reinterpret_cast<const int8_t*>(data + DWP_HEADER_BYTES);
    receive_output_aux<int8_t>(real_data, expected_dims, l);
  } else if (l->output_type == onnx::TensorProto_DataType_UINT8) {
    const uint8_t *real_data = reinterpret_cast<const uint8_t*>(data + DWP_HEADER_BYTES);
    receive_output_aux<uint8_t>(real_data, expected_dims, l);
  } else {
    log_fatal("can't compute with tensor of type {}\n", Op::get_tensorproto_dtype_name(l->output_type));
  }
}

HashedDispatchTable::HashedDispatchTable(const Fstream &fp) {
  const unsigned char *data = (const unsigned char *)fp.get_data();
  size_t size = fp.get_size();
  assert(size > DWP_HEADER_BYTES);
  uint32_t dwp_header = bytes2int(data);
  uint32_t ds = bytes2int(data + 4);
  uint32_t addr = bytes2int(data + 8);
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

