#include <stdio.h>
#include <numeric>
#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"

int main(int argc, char* argv[]){

	std::vector<float> expected { -371, -434, -497, -560, -623, -686, -749, 70, 56, 42, 28, 14, 0, -14, 511, 546, 581, 616, 651, 686, 721, 952, 1036, 1120, 1204, 1288, 1372, 1456, 1393, 1526, 1659, 1792, 1925, 2058, 2191, 1834, 2016, 2198, 2380, 2562, 2744, 2926, 2275, 2506, 2737, 2968, 3199, 3430, 3661 };

	const int SA_rows = 7;
	const int SA_columns = 7;
	const int input_rows = 7;
	const int input_columns = 7;

	std::vector<int> weight(SA_rows * SA_columns);
	std::vector<int> input_matrix(input_rows * input_columns);

	std::iota(weight.begin(), weight.end(), -12);
	std::iota(input_matrix.begin(), input_matrix.end(), -12);

	SA SA1(SA_rows, SA_columns);
	SA1.load_weights(weight);

	Chain c1;
	c1.push(new Chainblock());



	GemmTransformer GT1(input_rows,input_columns,SA_rows,SA_columns);
	auto out = GT1.transform(input_matrix);
	
	SA1.propagate(out,c1);
	SA1.print_array();
	auto t1 = SA1.get_output();
    auto computed = GT1.untransform(t1);
	Pooler p1(2,2,1,1);
	fMat pooler_output = p1.max_pooler(v2mat<int,int>(computed,7,7),7,7);
	std::vector<float> output =mat2v<float,float>(pooler_output,7,7);
    bool status = generate_report<float,float>(argv[0], expected, output );

	
    return status;


}