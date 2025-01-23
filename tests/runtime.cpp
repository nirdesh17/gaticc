#include <algorithm>
#include <cstdlib>       
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>      


std::unordered_map<std::string, std::pair<std::string, int>> model_map = {
    {"vgg_16_224_int8", {"235", 224}},
};

std::vector<std::string> summary;

void runtime() {
  std::string gaticc = std::getenv("GATICC_PATH");
  std::string test_model = "";
  if (std::getenv("TEST_MODEL") != nullptr) {
    test_model = std::getenv("TEST_MODEL");
  }

  std::vector<std::string> model_names;
  std::stringstream ss(test_model);
  std::string model;
  while (std::getline(ss, model, ',')) {
    model_names.push_back(model);
  }

  for (auto i : model_map) {

    if (!model_names.empty() &&
        std::find(model_names.begin(), model_names.end(), i.first) ==
            model_names.end()) {
      continue;
    }

    std::string onnx_file = gaticc + "model_zoo/models/" + i.first + ".onnx";
    std::string gml_file = gaticc + "model_zoo/models/" + i.first + ".gml";

    if (std::filesystem::exists(onnx_file)) {
      std::string compile = gaticc + "build/gaticc -c " + onnx_file + " -o " + gml_file + " " 
                            "--ramsize 512 --sa-arch 9,4,4 --vasize 32 "
                            "--accbuf-size 4096 --fcbuf-size 32768 ";

      std::system(compile.c_str());

      std::string command = "sudo " + gaticc + "build/gaticc -r " + gml_file + " "
                            "--run-onnx " + onnx_file + " "
                            "--inputpath " + gaticc + "dog.jpg "
                            "--size " + std::to_string(i.second.second) + " "
                            "--loadpy " + gaticc + "tests/runtime.py "
                            "--preprocfn preprocess "
                            "--postprocfn postprocess "
                            "--venv-path " + gaticc + "my_env/lib/python3.10/site-packages "
                            "--sa-arch 9,4,4 "
                            "--ramsize 512 "
                            "--vasize 32 "
                            "--accbuf-size 4096 "
                            "--fcbuf-size 32768 "
                            "--receive-over-uart 230400 ";

      std::system(command.c_str());

      std::ifstream temp_file("received_output.txt");
      std::string line;
      std::getline(temp_file, line);
      std::remove("received_output.txt");

      if (line != i.second.first) {
        summary.push_back(i.first + " failed");
      } else {
        summary.push_back(i.first + " passed");
      }
    } else {
      summary.push_back(i.first + " failed");
    }
  }

  for (auto i : summary) {
    std::cout << i << std::endl;
  }
}

int main() {

  runtime();
  return 0;
}