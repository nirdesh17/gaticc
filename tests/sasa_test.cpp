#include <numeric>
#include "../src/sim.h"
#include "../src/utils.h"
#include "../src/transformers.h"
#include "../src/sasa.h"

int main(){

	std::vector<std::vector<std::vector<int>>> input_tensor(1, std::vector<std::vector<int>>(7,std::vector<int>(7,10)));
	std::vector<std::vector<std::vector<std::vector<int>>>>  input_kernel(1,std::vector<std::vector<std::vector<int>>>(1,std::vector<std::vector<int>> (3,std::vector<int>(3,2))));
	Mat output;

	SASA s1(9,8,8);
	output = s1.master(input_tensor,input_kernel);
	print_vec_vec("output mat is : ", output);
}