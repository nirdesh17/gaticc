#include "options.h"
#include "executor.h"
#include "onnx_parser.h"
#include "rt.h"
#include "utils.h"
#include "optimization.h"
#include "instgen.h"
#include "archgen.h"

void dispatch_timeest(const Op::Parser &parser) {
  Op::Graph graph = parser.get_graph();
  time_estimate(graph);
}

void dispatch_info_ops() {
  std::string s = gbl_args["info"].as<std::string>();
  Op::Parser parser(s);
  if (gbl_args.has_option("summary")) {
    std::cout << "Input names: [";
    for (const auto &name : parser.get_model_input_names()) {
      std::cout << name << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Output names: [";
    for (const auto &name : parser.get_model_output_names()) {
      std::cout << name << ", ";
    }
    std::cout << "]" << std::endl;
    parser.bare_summary();
  }

  if (gbl_args.has_option("timeest")) {
    dispatch_timeest(parser);
  }

  if (gbl_args.has_option("flops")) {
    split_large_kernel(parser.get_graph());
    Pass::absorb(parser.get_graph());
    Op::Graph megablock_graph =
        Pass::create_megablock_graph(parser.get_graph());
    auto order = crt_exec_order(megablock_graph);
    std::vector<std::pair<std::string, long long>> flops =
        flops_estimate(order);
    for (const auto &[name, count] : flops) {
      std::cout << "FLOPs for " << name << ": " << count << " FLOPs"
                << std::endl;
    }
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
  log_fatal("command line driver disabled, use the python interface\n");
}

void dispatch_run_ops() {
  log_fatal("command line driver disabled, use the python interface\n");
}

void dispatch_archgen_ops() {
  std::string onnx_path = gbl_args["archgen"].as<std::string>();
  std::string fpga = get_fpga();
  Op::Parser parser(onnx_path);
  auto graph = parser.get_graph();
  auto arch = archgen(graph, fpga);
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
  } else if (gbl_args.has_option("archgen")) {
    dispatch_archgen_ops();
  } else {
    log_fatal("Don't know what to do. See gaticc -h\n");
  }
  return 0;
}
