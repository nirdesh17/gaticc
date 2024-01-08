#include <stdio.h>
#include <numeric>
#include "../src/sim.h"
#include "../src/transformers.h"
#include "../src/utils.h"

int main(int argc, char* argv[]){

	std::vector<float> expected { -42.4375, -104.125, -142.625, -128.625, 147.875, 306.25, 327.25, 257.25, 613.375, 1335.25, 1552.25, 1286.25, 721.875, 1580.25, 1853.25, 1543.5 };

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
	Mat temp_mat;



	GemmTransformer GT1(input_rows,input_columns,SA_rows,SA_columns);
	auto out = GT1.transform(input_matrix);
	
	SA1.propagate(out,c1);
	SA1.print_array();
	auto t1 = SA1.get_output();
    auto computed = GT1.untransform(t1);
	Pooler p1;
	temp_mat= v2mat<int,int>(computed,7,7);
	fMat pooler_output = p1.average_pooler(temp_mat,7,7,2,2,1,4,4);
	std::vector<float> output;
	output = mat2v<float,float>(pooler_output,7,7);
    bool status = generate_report<float,float>(argv[0], expected, output );

	
    return status;


}