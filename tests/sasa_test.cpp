#include "../src/sasa.h"
#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"
#include <chrono>
#include <numeric>
#include <stdlib.h>

int main() {

  std::vector<std::vector<std::vector<int>>> input_tensor(
      64, std::vector<std::vector<int>>(224, std::vector<int>(224, 10)));
  std::vector<std::vector<std::vector<std::vector<int>>>> input_kernel(
      32, std::vector<std::vector<std::vector<int>>>(
              64, std::vector<std::vector<int>>(3, std::vector<int>(3, 2))));
  std::vector<std::vector<std::vector<std::vector<int>>>> input_kernel_2(
      64, std::vector<std::vector<std::vector<int>>>(
              32, std::vector<std::vector<int>>(3, std::vector<int>(3, 2))));

  std::vector<Mat> output;
  auto start = std::chrono::high_resolution_clock::now();
  SASA s1(9, 8, 8);
  output = s1.master(input_tensor, input_kernel);
  output = s1.master(output,input_kernel_2);
  std::cout << "output kernel size : " << output.size() << std::endl;
  std::cout << "output row size : " << output.at(0).size() << std::endl;
  std::cout << "output col size : " << output.at(0).at(0).size() << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(stop - start);

  std::cout <<"time taken by the whole pgm "<< duration.count() <<" sec"<< std::endl;

  return 0 ;  

}