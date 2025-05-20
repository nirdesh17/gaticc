#include "options.h"
#include "gati.h"
#include "utils.h"

int main(int argc, char *argv[]) {
  gbl_args.parse(argc, argv);
  init();
  dispatch();
}
