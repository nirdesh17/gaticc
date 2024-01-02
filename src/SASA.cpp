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



//we can still continue with our approach of pointer...
// incoming dims [kernels, channels, k_row,k_column]
class SASA{

	void create(int channels,int rows,int columns,int channel, int filter);
	
};

void SASA::create(int channels,int rows,int columns, int channel, int filter){

	// SA* SA_ptr= (SA*)malloc(sizeof(SA)*columns*rows*channels);
	// SA* new_SA_ptr[ channel*filter];
	std::vector<SA*> new_SA_ptr;  // new_SA_ptr(8); better than an array bc once returned from fucntion it will cleared // these indexes are channel numbers

	// std::vector<SA*> R_SA_ptr;
	// std::vector<SA*> G_SA_ptr;
	// std::vector<SA*> B_SA_ptr;
	// R_SA_ptr.push_back((SA*)malloc(sizeof(SA)));   // K times each 
	// G_SA_ptr.push_back((SA*)malloc(sizeof(SA)));
	// B_SA_ptr.push_back((SA*)malloc(sizeof(SA)));   

	// func to call each constructor of the SA
	// for(int i = 0 ; i < k_max ; i++){
	// 	R_SA_ptr.at(i)->(rows,columns);
	// 	G_SA_ptr.at(i)->(rows,columns);
	// 	B_SA_ptr.at(i)->(rows,columns);
	// }
	for(int i = 0 ; i < k_max ; i++){
		// new_SA_ptr.insert(i,(SA*)malloc(sizeof(SA(9,8))));  // contructor would be called here if n
		new_SA_ptr.push_back((SA*)malloc(sizeof(SA(9,8))));   // we have to keep this dynamic bc if sometimes kernels are less than 8 or channels are less than 8 in that case there will be empty columns and matrices
		
		if(i>=8 &&<16){
			new_SA_ptr.push_back((SA* ptr = new_SA_ptr.at(0)));  // ptr pointing to the same container
		}
		
		
		// new_SA_ptr.at(i)->SA(rows,columns); this is illegal          // now the issue is if we keep on loading kernels and at some point it ended in the middle of the systolic array then at that moment ? ig we can put zeroes ?
		// SA_list.push_back(*(new SA(9,8)));									// 
	}
	
	}

	// for(int i = 0 ; i < k_max ; i ++){
	// R_SA_ptr.at(i)->load_weights();
	// G_SA_ptr.at(i)->load_weights();
	// B_SA_ptr.at(i)->load_weights();

	// }
	

	std::vector<fMat> kernel_tensor_vals;   // index of this matrix is increasing order of channels + filters


/*Loading kernel tensor values in SAs*/
	

	// for(int i = row_count_input ; i < k_max ; i++){
	// 	for(int j = column_count_input ; j < ch ; j ++ )
	// 	new_SA_ptr.at(i*ch + j)->load_weights(mat2v<float,float>(kernel_tensor_vals.at(i*ch + j)));  // also check how its stored 1D or 2D 
	// 	column_count_input++;	
	// }

	//input matrix with channels // number of channels = ch
	

	// std::vector<ConvTransformer*> CT_ptr(ch);
	// for(int i = 0; i< ch; i++ ){
	// 	CT_ptr.push_back((ConvTransformer*)malloc(sizeof(ConvTransformer)));

	// 	std::vector<Mat> transformed_input_mats.at(i).push_back( CT_ptr.at(i)->transform(/*intput matrix tensors*/));
	// }

	// now we will perform operation between inputs and SA

	// for(int i = 0 ; i < k_max ; i++){

	// 	// R_SA_ptr.at(i)->propagate(transformed_input_mats.at(i));
	// 	// G_SA_ptr.at(i)->propagate(transformed_input_mats.at(i+1));
	// 	// G_SA_ptr.at(i)->propagate(transformed_input_mats.at(i+2));

	// 	// _propagate pushed to output array and after propagate we can just call get out array
	// 	std::vector<Mat> final_mats;   // expecting input of also this form
	// 	final_mats.at(i).push_back(  R_SA_ptr.at(i)->get_output() + G_SA_ptr.at(i)->get_output() + B_SA_ptr.at(i)-> get_output);

	// 	// _propagte should return single element vals and we'll add them at the spot change the above with the mixture of it
	// 	// pnce we get that we will get 1 matrix out of these three channels and then repeating the process will give back total matrices 

	// 	// we have to create buffer to store the tensor vals temporarily but it should be created once and used again and again
	// 	// because the input is recieved once and after that all the other values are just processed values

	// }

 std::vector<fMat> input_matrix_transformer(std::vector<fMat> input){   // can be made to give out matrix out one by one instead of the whole bunch

	ConvTransformer* ip_CT_ptr    //[input.size()];  // will return number of channels /rows in the dim of 1xChxRxCol
	// transformer for each channel of input_tensor
	std::vector<fMat> transformed_mats;

	for(int i = 0 ; i < input.size() ; i ++){
		tansformed_mats.at(i).push_back( ip_CT_ptr[i]->transform(mat2v<float,float>(input.at(i))));    // creating a system of matrices which can push elements to all the K_max SAs
	}
	return transformed_mats;
}

// ALL OF THIS IS A BLUNDER ... IN ORDER TO PUT IN WEIGHTS LIKE SA DOES WE NEED TO BE ITS FRIEND CLASS SO WE CAN ACCESS PRIVATE MEMBERS 


void weight_loader(){
int row_count_input=0;
	int column_count_input=0;
	for(int i = 0 ; i < fpga_channel; i ++ ) {
		for(int j = 0 ;j< SA_columns;j++){

			if(SA_columns> input_kernels){
				break;
			}
		new_SA_ptr.at(i)->load_weights_new(input.at(j).at(i),position_j)   // input is 4D - input.at(kernel).at(channel)
	}
}
}
void operator(std::vector<fMat> transformed_mats, std::vector<SA*> Systolic_arrays){  // SA pointer array should be passed instead of vector?

	// indexes in the multiples of channels will be the 3D SAs

	std::vector<fMat> adder_mat;
	
	for(int i = 0 ; i < fpga_channels; i ++){
			// yaha pr b wo dekhna padega jo limited jaga h fpga pr aur phir reuse krte h
		
			Systolic_array.at(i)->propagate(transformed_mats.at(i));
			// get output matrix here and store
			adder_mat.at(i)= v2mat<float,float>(get_output());  //224*224 elements will be in 1 column so we have to think about it

		
	}

	//return to controller so it can reload weights
}

void controller(){

	for(int i = 0 ; i < ; i ++){
	weight_loader;
	operator();

	finalmat;
	}

}




	
	
	
	

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


void SA::load_weights_new(/*std::vector<int>& weights*/ fMat kernel_weight, int j) {  // template for floats also   will be called like load_weight_new(input.at(i).at(j)))
    
	std::vector<float> temp_vec;
	temp_vec=mat2v<float,float>(kernel_weight,3,3);   // 3 x 3 is fixed for now , but we have to make it flexible
	for (int i = 0; i < SA_rows /*9*/; ++i) {
        get_pe_from_vertex(vertarray[j+ i*SA_columns /*8*/]).set_weight(/*weights[i]*/ temp_vec[i] );   //these are full kernels
    }
}


// the systolic array is a 2D graph with vertices as linear and increases horizontally , graph manages the connection itself
// so in order to fill the weights column wise. we have to fill the weights using vertarray / rowsize and then transpose.
// ig we have to break the load weights func after the number of columns and then repeat it and then finally in the end transpose it as a whole

// propagate and _propagate works fine , it all depends on how we load our weights in our SAs

}


int main(){

		
}



// TODO for tomorrow:
// 1.find a way to save the data comming out of SA and adding it with different channels
// 2.how to reload SAs  in void controller . it will watch over everything