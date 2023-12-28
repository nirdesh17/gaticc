#include <vector>
#include <numeric>
#include <utility>
#include <algorithm>
#include <iostream>

#include "sim.h"
#include "transformers.h"
#include "utils.h"

int k_max; // max number of SA in FPGA 
int ch; // max number of input channels
int filters_max; // input channels or node demanded channel number
class SASA{

	void create(int channels,int rows,int columns,int channel, int filter);
	
};

void SASA::create(int channels,int rows,int columns, int channel, int filter){

	SA* SA_ptr= (SA*)malloc(sizeof(SA)*columns*rows*channels);
	// SA* new_SA_ptr[ channel*filter];
	std::vector<SA*> new_SA_ptr(channel*filter);  // better than an array bc once returned from fucntion it will cleared

	std::vector<SA*> R_SA_ptr;
	std::vector<SA*> G_SA_ptr;
	std::vector<SA*> B_SA_ptr;
	R_SA_ptr.push_back((SA*)malloc(sizeof(SA)));   // K times each 
	G_SA_ptr.push_back((SA*)malloc(sizeof(SA)));
	B_SA_ptr.push_back((SA*)malloc(sizeof(SA)));   

	// func to call each constructor of the SA
	// for(int i = 0 ; i < k_max ; i++){
	// 	R_SA_ptr.at(i)->(rows,columns);
	// 	G_SA_ptr.at(i)->(rows,columns);
	// 	B_SA_ptr.at(i)->(rows,columns);
	// }
	for(int i = 0 ; i < k_max ; i++){
		new_SA_ptr.insert(i,(SA*)malloc(sizeof(SA)*columns*rows*channels));  // contructor would be called here if n
		// new_SA_ptr.at(i)->SA(rows,columns); this is illegal 
	}

	// for(int i = 0 ; i < k_max ; i ++){
	// R_SA_ptr.at(i)->load_weights();
	// G_SA_ptr.at(i)->load_weights();
	// B_SA_ptr.at(i)->load_weights();

	// }
	

	std::vector<fMat> kernel_tensor_vals;   // index of this matrix is increasing order of channels + filters


/*Loading kernel tensor values in SAs*/


	for(int i = 0 ; i < k_max ; i++){
		for(int j = 0 ; j < ch ; j ++ )
		new_SA_ptr.at(i*ch + j)->load_weights(mat2v<float,float>(kernel_tensor_vals.at(i*ch + j)));  // also check how its stored 1D or 2D 
	}

	//input matrix with channels // number of channels = ch
	

	std::vector<ConvTransformer*> CT_ptr(ch);
	for(int i = 0; i< ch; i++ ){
		CT_ptr.push_back((ConvTransformer*)malloc(sizeof(ConvTransformer)));

		std::vector<Mat> transformed_input_mats.at(i).push_back( CT_ptr.at(i)->transform(/*intput matrix tensors*/));
	}

	// now we will perform operation between inputs and SA

	for(int i = 0 ; i < k_max ; i++){

		R_SA_ptr.at(i)->propagate(transformed_input_mats.at(i));
		G_SA_ptr.at(i)->propagate(transformed_input_mats.at(i+1));
		G_SA_ptr.at(i)->propagate(transformed_input_mats.at(i+2));

		// _propagate pushed to output array and after propagate we can just call get out array
		std::vector<Mat> final_mats;   // expecting input of also this form
		final_mats.at(i).push_back(  R_SA_ptr.at(i)->get_output() + G_SA_ptr.at(i)->get_output() + B_SA_ptr.at(i)-> get_output);

		// _propagte should return single element vals and we'll add them at the spot change the above with the mixture of it
		// pnce we get that we will get 1 matrix out of these three channels and then repeating the process will give back total matrices 

		// we have to create buffer to store the tensor vals temporarily but it should be created once and used again and again
		// because the input is recieved once and after that all the other values are just processed values

	}

 std::vector<fMat> input_matrix_transformer(std::vector<fMat> input){   // can be made to give out matrix out one by one instead of the whole bunch

	ConvTransformer* ip_CT_ptr[input.size()];  // will return number of channels /rows
	std::vector<fMat> transformed_mats;

	for(int i = 0 ; i < input.size() ; i ++){
		tansformed_mats.at(i).push_back( ip_CT_ptr[i]->transform(mat2v<float,float>(input.at(i))));    // creating a system of matrices which can push elements to all the K_max SAs
	}
	return transformed_mats;
}

void operator(std::vector<fMAt> transformed_mats, std::vector<SA*> Systolic_arrays){  // SA pointer array should be passed instead of vector?

	// indexes in the multiples of channels will be the 3D SAs

	std::vector<fMat> final_mats;
	std::vector<fMat> intermediate_mats;

	for(int i =0 ; i < k_max; i++){   // if k max_ < i_max   i < (k_max< i_max?kmax:i_max)
		for(int j =0 ; j < ch /* or transformed matrix size */; j++){
		Systolic_arrays.at(i*ch+j)->propagate(transformed_mats.at(j));// clock cycle issue 
		intermediate_mats.at(i*ch +j) = Systolic_arrays.at(i*ch+j)->get_output();
		}

		// add here the internal vals

		final_mats.at(i).push_back(add_mat_func(intermediate_mats,ch,i)); // storing the final matrices  , here at at(i) 'i' will cause problem for the next time fucntion is called for continue for i_max minus k_max
	} // and then finally activating chainblock
}


std::vector<fMat> add_mat_func(std::vector<fMat> intermediate_mats, int channels,int index){   // return value problem
	std::vector<fMat> final_mats;

	for(int i =index ; i < channels; i ++){
		if(i==0){
	final_mats.at(index).push_back(add_mat<float,float>(intermediate_mats.at(i),intermediate_mats.at(i+1)));
		}
	final_mats.at(index).push_back(add_mat<float,float>(final_mats.at(index),intermediate_mats.at(i+2)));
	}

	return final_mats;
}



}


int main(){


}