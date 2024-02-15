#include <iostream>
#include <numeric>

#include "ffi.h"
#include "Python.h"
#include "onnx_parser.h"
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

  if (gbl_args.has_option("onnx")) {
    std::string s = gbl_args["onnx"].as<std::string>();
    Op::Parser parser(s);
    if (gbl_args.has_option("timeest")) {
      std::string arch_list = gbl_args["timeest"].as<std::string>();
      std::vector<int> mnk = parse_csv_string(arch_list);
      assert(mnk.size() != 0 && "Ill formatted dimension string to --timeest, expects string like 9,8,8");
      assert(mnk.size() == 3 && "Systolic Array shape should be 3 dimensional M, N, K");
      parser.time_estimate(mnk.at(0), mnk.at(1), mnk.at(2));
    } else if (gbl_args.has_option("summary")) {
      parser.summary();
    } else {
      gbl_args.print_usage();
      log_fatal("Do not know what to do with --onnx, specify atleast one operation like --summary or --timeest");
    }
  }
}
