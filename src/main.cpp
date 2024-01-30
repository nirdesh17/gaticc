#include <iostream>
#include <numeric>

#include "argagg.h"
#include "ffi.h"
#include "onnx_parser.h"
#include "ops.h"
#include "sim.h"
#include "transformers.h"
#include "utils.h"

class Argparse {
  const char *usage = "Usage: sysim [OPTIONS]\n";
  argagg::parser_results args;
  argagg::parser argparser{
      {{"help", {"-h", "--help"}, "get this help message nigga", 0},
       {"verbose", {"-v", "--v"}, "verbose", 0},
       {"onnx", {"--onnx"}, "load onnx file", 1}}};

public:
  Argparse(int argc, char *argv[]) {
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

  void print_usage() const {
    std::cerr << usage << argparser;
  }
};

int main(int argc, char *argv[]) {
  Argparse args(argc, argv);

  if (args.has_option("help")) {
    args.print_usage();
  }

  PyEngine engine("src.preprocess");
  std::vector<int> img = py_read_img(engine, std::string("images/mug.jpg"));
  //print_vec<int>("img", img);

  if (args.has_option("onnx")) {
    std::string s = args["onnx"].as<std::string>();
    Op::Parser parser(s);
    parser.get_execution_order();
    parser.time_estimate(9, 8, 8);
  }
}
