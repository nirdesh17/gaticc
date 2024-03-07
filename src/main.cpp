#include "ffi.h"
#include "utils.h"
#include <filesystem>
#include "options.h"

/* instance of the gbl_args extern declaration in utils.h */
Argparse gbl_args;


int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);

  if (gbl_args.has_option("help")) {
    gbl_args.print_usage();
    std::exit(EXIT_SUCCESS);
  }

  std::filesystem::path mod_path("src/");
  PyEngine engine("ml_inference", mod_path);

  if (gbl_args.has_option("onnx")) {
    dispatch_onnx_ops(engine);
  }
}
