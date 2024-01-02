#include <iostream>
#include "sim.h"
#include "transformers.h"
#include "utils.h"


// we can only keep master in public and keep all the other functions inside private
class SASA{  // for now its fixed in FPGA = ( 9 x 8 ) x 8 but class is flexible
	private:

		int sa_channels;
		int sa_channel_rows;
		int sa_channel_columns;
		std::vector<SA*> SA_ptr;
		// std::vector<ConvTransformer*> ip_CT_ptr;

	public:

		SASA(int sa_channel_rows,int sa_channel_columns,int sa_channels);
		// ~SASA();
		std::vector<SA*> create_sasa(int sa_channel_rows,int sa_channel_columns, int sa_channels);
		std::vector<ConvTransformer*> create_ConvTransformer(int IW, int IH, int KW, int KH, int srows, int scols){

		std::vector<fMat> input_tensor_transformer(std::vector<fMat> input_tensor);  
		std::vector<float> SASA:: load_kernel_tensors(std::vector<SA*> SA_ptr,std::vector<std::vector<fMat>> input_kernel, int kernel_channel, int kernel_number); // dims: kernel x channels x row x column , index (here) : kernel
		void load_weights_tensor(fMat kernel_weight, int column_pos);
		void master(std::vector<fMat> input_tensor,std::vector<std::vector<fMat>> input_kernel);
		void operatorr(std::vector<fMat> transformed_mats, std::vector<SA*> SA_ptr);
	
};

SASA::SASA(int sa_channel_rows,int sa_channel_columns,int sa_channels): sa_channel_rows{sa_channel_rows},
	  sa_channel_columns{sa_channel_columns}, sa_channels{sa_channels}{}

std::vector<SA*> SASA::create_sasa(int sa_channel_rows,int sa_channel_columns, int sa_channels){
	load_kernel_tensors(std::vector<SA*> SA_ptr,std::vector<std::vector<fMat>> input_kernel, int kernel_channel); // dims: kernel x channels x row x column , index (here) : kernel
		void load_weights_tensor(fMat kernel_weight, int column_pos);
	
	std::vector<SA*> SA_ptr;

	for(int i = 0 ; i < sa_channels ; i++){

		SA_ptr.push_back((SA*)malloc(sizeof(SA(sa_channel_rows,sa_channel_columns))));}

		return SA_ptr;
}

std::vector<ConvTransformer*> SASA::create_ConvTransformer(int IW, int IH, int KW, int KH, int srows, int scols){
	std::vector<ConvTransformer*> CT_ptr;

	for(int i = 0 ; i < sa_channels ; i++){

		CT_ptr.push_back((ConvTransformer*)malloc(sizeof(ConvTransformer(IW,IH,KW,KH,srows,scols))));}

		return CT_ptr;
}



std::vector<fMat> SASA::input_tensor_transformer(std::vector<fMat> input_tensor){   // input_tensor dims: channel x rows x columns , index: channel

	std::vector<ConvTransformer*> ip_CT_ptr(input_tensor.size());  

	std::vector<fMat> transformed_mats(input_tensor.size());

	for(int i = 0 ; i < input_tensor.size() ; i ++){
		transformed_mats.at(i).push_back( ip_CT_ptr.at(i)->(fMat)transform(mat2v<float,float>(input_tensor.at(i), input_tensor.at(i).size(),input_tensor.at(i).at(i).size()))); 
	}
	return transformed_mats;

}

std::vector<float> SASA:: load_kernel_tensors(std::vector<SA*> SA_ptr,std::vector<std::vector<fMat>> input_kernel, int kernel_channel,int kernel_number){



	std::vector<float> kernel_tensor(sa_channel_rows*sa_channel_columns);
	std::vector<float> output_tensor(sa_channel_rows*sa_channel_columns);


	for(int i = kernel_number ; i < (kernel_number+ sa_channel_columns) ; i ++){  // for now assuming sa_channel_columns < input_kernel.size()
		
		kernel_tensor = mat2v<float,float>(input_kernel.at(i).at(kernel_channel), input_kernel.at(0).at(0).size(),input_kernel.at(0).at(0).at(0).size());

		for(int j = 0 ; j < sa_channel_rows; j++){
		
			output_tensor.push_back(kernel_tensor[j]);
		}
		
	}

	return output_tensor;
	


	// int row_count_input=0;  
	// int column_count_input=0;
	// for(int i = 0 ; i < sa_channels; i ++ ) {
	// 	for(int j = 0 ;j< sa_channel_columns;j++){    // working for 1st go

	// 		// if(SA_columns> input_kernels){
	// 		// 	break;
	// 		// }
	// 	load_weights_tensor(SA_ptr.at(i),input_kernel.at(j).at(i),j)  ; // input is 4D - input.at(kernel).at(channel)
	// 	}
	// }
}


void SASA::load_weights_tensor(SA* SA_ptr, ConvTransformer CT, std::vector<float> input/*fMat kernel_weight, int column_pos*/) {  // template for floats also   will be called like load_weight_new(input.at(i).at(j)))
    // transform weights is a func of Convtransform
	SA_ptr->load_weights(CT.transform_weights(input,sa_channel_rows,sa_channel_columns));
	
}


void SASA::operatorr(fMat transformed_mats, SA* SA_ptr){  // SA pointer array should be passed instead of vector?

	// indexes in the multiples of channels will be the 3D SAs

	std::vector<fMat> adder_mat(8); 
	std::vector<fMat> output;
	std::vector<float> vec;
	
	for(int i = 0 ; i < sa_channels; i ++){
			// yaha pr b wo dekhna padega jo limited jaga h fpga pr aur phir reuse krte h
		
			SA_ptr->propagate(transformed_mats);
			output = SA_ptr->get_output(); 
			vec = CT1.untransform(output);



			
	}

	return 

}

/* the input is in the form of : channel-> kernel -> elements
   the output will have the output stored in only one channel(0) having the total number of kernels ... that channel will have no significance 
   of its index .
*/

std::vector<std::vector<float>> SASA:: adder(std::vector<fMat> input){


	for(int m = 0 ; m < ; m ++){      		// m < input_kernel.size() = # of kernels
			for(int n = 0 ; n < ; n ++){     // rotating channels.

				for(int p = 0 ; p < ; p ++){

				

				input.at(0).at(m).at(p)= input.at(0 /*kernel*/).at(m /*channel*/).at(p) + input.at(n+1).at(m).at(p)

				// out_vec.at(m).at(n).push_back = vec.at(m).at(n) + vec.at(m+1).at(n)
				// 								+vec.at(m+2).at(n)+ vec.at(m+3).at(n)
				// 								+vec.at(m+4).at(n)+ vec.at(m+5).at(n)
				// 								+vec.at(m+6).at(n)+ vec.at(m+7).at(n);
				}
			}
			return input.at(0);
}



void SASA:: master(std::vector<fMat> input_tensor,std::vector<std::vector<fMat>> input_kernel){          // master/ Control Unit
	
	std::vector<SA*> SA_ptr = create_sasa(sa_channel_rows,sa_channel_columns,sa_channels);

	std::vector<fMat> transformed_mats;
	std::vector<float> output_weights;
	std::vector<std::vector<fMat>> vec;   // channel pointers -> kernel pointers -> C1K1
	std::vector<std::vector<float>> out_vec;
	std::vector<ConvTransformer*> CT_ptr= create_ConvTransformer();  // call its contructor

	// ConvTransformer CT1;

	transformed_mats = input_tensor_transformer(input_tensor);
/* master is filling the SAs kernel wise ... filling up to its capacity and then reloading the
*  SAs with same channels but different kernel and then after all the kernels are done ....
* channel is reloaded and then the process is done all over again.
*/

for(int k = 0 ;  k < input_kernel.at(0).size(); k ++ ){  // for reloading SAs   // k will rotate kernel channels
	for( int i = 0 ; i < input_kernel.size(); i ++){					// i is rotating kernels 
		for( int j = 0 ; j < sa_channels; j++){
		
			output_weights= load_kernel_tensors(SA_ptr.at(j),input_kernel,k*sa_channels +j, i*8);     // assuming sa_channels < input_kernel channels                       // Load Kernel to each channel in the
			load_weights_tensor(SA_ptr.at(j), CT_ptr.at(j), output_weights);  // i think these should complete all the SA channels in one go so we can reload them easily
			vec.at(k).at(i*sa_channel_columns + j).push_back(operatorr(transformed_mats.at(j),SA_ptr.at(j)));
			
		}

		
		}

		
	}

	// add here   
	adder(vec);
}








// so if the kernel is 1,2,3...,9 and the other kernel which is 10,...,18 and last one is 19,,, 27 and 
// we load all of them in a single vector starting from 1,2,3...,27 and call transform weightsa(9,3) on it .,,, it will
// arrange them in a 9x3  SA with first column as 1,2,3,4...,9 
// so now we have to get a single vector concatinate it with c0k0 then c0k1 then c0k2 upto c0k7 and call transform_weight on it and then it will arrange it in the order we WANT