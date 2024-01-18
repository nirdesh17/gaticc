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
  Mat output;
  // clock_t timer = clock();
  auto start = std::chrono::high_resolution_clock::now();
  SASA s1(9, 8, 8);
  output = s1.master(input_tensor, input_kernel);
  std::cout << "output row size : " << output.size() << std::endl;
  std::cout << "output col size : " << output.at(0).size() << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(stop - start);

  std::cout <<"time taken by the whole pgm "<< duration.count() <<" sec"<< std::endl;
  
//   sleep(10000);


}