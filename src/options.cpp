#include "options.h"
#include "executor.h"
#include "ffi.h"
#include "onnx_parser.h"
#include "rt.h"
#include "utils.h"
#include "optimization.h"
// #include <cassert>
// #include <string>
// #include <vector>
#include "instgen.h"

void dispatch_simulator(const Op::Parser &parser) {
  if (!gbl_args.has_option("loadpy")) {
    log_fatal("Option --loadpy needs to be specified for simulation\n");
    gbl_args.print_usage();
  }

  if (!gbl_args.has_option("preprocfn")) {
    log_fatal("Option --preprocfn needs to be specified for simulation\n");
    gbl_args.print_usage();
  }

  if (!gbl_args.has_option("postprocfn")) {
    log_fatal("Option --postprocfn needs to be specified for simulation\n");
    gbl_args.print_usage();
  }
  std::string mod_arg = gbl_args["loadpy"].as<std::string>();
  std::string mod_name = extract_basename(mod_arg).stem().string();
  auto mod_path = extract_dirname(mod_arg);

  PyEngine engine(mod_name, mod_path);
  Executor e(engine, parser);
}

void dispatch_timeest(const Op::Parser &parser) {
  Op::Graph graph = parser.get_graph();
  time_estimate(graph);
}

void dispatch_info_ops() {
  std::string s = gbl_args["info"].as<std::string>();
  Op::Parser parser(s);
  if (gbl_args.has_option("summary")) {
    parser.bare_summary();
  }

  if (gbl_args.has_option("timeest")) {
    dispatch_timeest(parser);
  }
}

void dispatch_compile_ops() {
  std::string s = gbl_args["compile"].as<std::string>();
  Op::Parser parser(s);
  split_large_kernel(parser.get_graph());
  Pass::absorb(parser.get_graph());
  GmlGen gmlgen(GATI_INST_ORG);
  BinBlob binblob{gmlgen.generate_gml(parser)};

  if (gbl_args.has_option("output")) {
    auto filename = gbl_args["output"].as<std::string>();
    binblob.write(filename);
  }

  if (gbl_args.has_option("pretty-print-blob")) {
    binblob.pretty_print();
  }
}

void dispatch_sim_ops() {
  std::string s = gbl_args["sim"].as<std::string>();
  Op::Parser parser(s);
  split_large_kernel(parser.get_graph());
  dispatch_simulator(parser);
}

void dispatch_run_ops() {
  if (!gbl_args.has_option("run_onnx")) {
    log_fatal(
        "couldn't find onnx file. Use --run-onnx to provide one or see help\n");
  }
  auto onnx_file = gbl_args["run_onnx"].as<std::string>();
  std::cout << "onnx file2 " << onnx_file << '\n';
  Op::Parser parser(onnx_file);
  split_large_kernel(parser.get_graph());
  Pass::absorb(parser.get_graph());
  Runner runner(parser);
}

int dispatch() {
  if (gbl_args.has_option("help")) {
    gbl_args.print_usage();
    return 0;
  } else if (gbl_args.has_option("version")) {
    gbl_args.print_version();
    return 0;
  } else if (gbl_args.has_option("info")) {
    dispatch_info_ops();
  } else if (gbl_args.has_option("compile")) {
    dispatch_compile_ops();
  } else if (gbl_args.has_option("sim")) {
    dispatch_sim_ops();
  } else if (gbl_args.has_option("run")) {
    dispatch_run_ops();
  } else {
    log_fatal("Don't know what to do. See gaticc -h\n");
  }
  return 0;
}
