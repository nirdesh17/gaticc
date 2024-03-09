#include "executor.h"
#include "ffi.h"
#include "onnx_parser.h"
#include "options.h"
#include "utils.h"
#include <cassert>
#include <string>
#include <vector>

void dispatch_onnx_ops(PyEngine &engine) {
  std::string s = gbl_args["onnx"].as<std::string>();
  Op::Parser parser(s);
  if (gbl_args.has_option("timeest")) {
    std::string arch_list = gbl_args["timeest"].as<std::string>();
    std::vector<int> mnk = parse_csv_string<int>(arch_list);
    assert(mnk.size() != 0 && "Ill formatted dimension string to --timeest, "
                              "expects string like 9,8,8");
    assert(mnk.size() == 3 &&
           "Systolic Array shape should be 3 dimensional M, N, K");
    parser.time_estimate(mnk.at(0), mnk.at(1), mnk.at(2));
  } else if (gbl_args.has_option("summary")) {
    parser.bare_summary();
  } else if (gbl_args.has_option("sim")) {
    std::string img_path = gbl_args["sim"].as<std::string>();
    Executor e(engine, parser, img_path);
  } else {
    gbl_args.print_usage();
    log_fatal("Do not know what to do with --onnx, specify atleast one "
              "operation like --summary or --timeest");
  }
}
