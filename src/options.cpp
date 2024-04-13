#include "executor.h"
#include "ffi.h"
#include "onnx_parser.h"
#include "options.h"
#include "utils.h"
#include <cassert>
#include <string>
#include <vector>

void dispatch_simulator(const Op::Parser &parser) {
  if (!gbl_args.has_option("loadpy")) {
    log_fatal("Option --loadpy needs to be specified for simulation");
    gbl_args.print_usage();
  }

  if (!gbl_args.has_option("preprocfn")) {
    log_fatal("Option --preprocfn needs to be specified for simulation");
    gbl_args.print_usage();
  }

  if (!gbl_args.has_option("postprocfn")) {
    log_fatal("Option --postprocfn needs to be specified for simulation");
    gbl_args.print_usage();
  }
  std::string mod_arg = gbl_args["loadpy"].as<std::string>();
  std::string mod_name = extract_basename(mod_arg).stem().string();
  std::filesystem::path mod_path = extract_dirname(mod_arg);

  PyEngine engine(mod_name, mod_path);
  Executor e(engine, parser);
}

void dispatch_timeest(const Op::Parser &parser) {
  std::string arch_list = gbl_args["timeest"].as<std::string>();
  std::vector<int> mnk = parse_csv_string<int>(arch_list);
  assert(mnk.size() != 0 && "Ill formatted dimension string to --timeest, "
                            "expects string like 9,8,8");
  assert(mnk.size() == 3 &&
         "Systolic Array shape should be 3 dimensional M, N, K");
  parser.time_estimate(mnk.at(0), mnk.at(1), mnk.at(2));
}

void dispatch_onnx_ops() {
  std::string s = gbl_args["onnx"].as<std::string>();
  Op::Parser parser(s);
  if (gbl_args.has_option("timeest")) {
    dispatch_timeest(parser);
  } else if (gbl_args.has_option("summary")) {
    parser.bare_summary();
  } else if (gbl_args.has_option("sim")) {
    dispatch_simulator(parser);
  } else {
    gbl_args.print_usage();
    log_fatal("Do not know what to do with --onnx, specify atleast one "
              "operation like --summary or --timeest");
  }
}

