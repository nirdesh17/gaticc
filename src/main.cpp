#include <iostream>
#include <numeric>

#include "ffi.h"
#include "Python.h"
#include "onnx_parser.h"
#include "ops.h"
#include "sim.h"
#include "transformers.h"
#include "utils.h"

/* instance of the gbl_args extern declaration in utils.h */
Argparse gbl_args;

int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);

  if (gbl_args.has_option("help")) {
    gbl_args.print_usage();
    std::exit(EXIT_SUCCESS);
  }

  // PyEngine engine("src.preprocess");
  // std::vector<int> img = py_read_img(engine, std::string("images/mug.jpg"));
  //print_vec<int>("img", img);

  if (gbl_args.has_option("onnx")) {
    std::string s = gbl_args["onnx"].as<std::string>();
    Op::Parser parser(s);
    parser.get_execution_order();
    parser.time_estimate(9, 8, 8);
  }
}
