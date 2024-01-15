#include <numeric>
#include "../src/sim.h"
#include "../src/utils.h"
#include "../src/transformers.h"
#include "../src/sasa.h"
#include <time.h>
#include <chrono>

int main(){

	std::vector<std::vector<std::vector<int>>> input_tensor(24, std::vector<std::vector<int>>(224,std::vector<int>(224,10)));
	std::vector<std::vector<std::vector<std::vector<int>>>>  input_kernel(32,std::vector<std::vector<std::vector<int>>>(24,std::vector<std::vector<int>> (3,std::vector<int>(3,2))));
	Mat output;
	// clock_t timer = clock();
	auto start = std::chrono::high_resolution_clock::now();
	SASA s1(9,8,8,true);
	output = s1.master(input_tensor,input_kernel);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
 
	std::cout << duration.count() << std::endl;

}