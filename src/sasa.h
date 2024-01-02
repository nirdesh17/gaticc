#include <iostream>
#include "sim.h"
#include "transformers.h"
#include "utils.h"

class SASA{
	private:

		int sa_channels;
		int sa_channel_rows;
		int sa_channel_columns;
		std::vector<SA*> SA_ptr;

	public:

		SASA(int sa_channel_rows,int sa_channel_columns,int sa_channels);
		// ~SASA();
		std::vector<SA*> create_sasa(int sa_channel_rows,int sa_channel_columns, int sa_channels);
		std::vector<ConvTransformer*> create_ConvTransformer(int IW, int IH, int KW, int KH, int srows, int scols){

		std::vector<fMat> input_tensor_transformer(std::vector<fMat> input_tensor, std::vector<ConvTransformer*> CT_ptr);  
		std::vector<float> load_kernel_tensors(std::vector<std::vector<fMat>> input_kernel, int kernel_channel, int kernel_number); // NCHW
		void load_weights_tensor(fMat kernel_weight, int column_pos);
		void master(std::vector<fMat> input_tensor,std::vector<std::vector<fMat>> input_kernel);
		fMat operatorr(std::vector<fMat> transformed_mats, std::vector<SA*> SA_ptr, ConvTransformer* CT_ptr);
		void SASA::splitter(std::vector<std::vector<fMat>>&vec , fMat temp_mat , int channel_number);
		fMat SASA:: adder(std::vector<fMat> input);
	
};

SASA::SASA(int sa_channel_rows,int sa_channel_columns,int sa_channels): sa_channel_rows{sa_channel_rows},
	  sa_channel_columns{sa_channel_columns}, sa_channels{sa_channels}{}

std::vector<SA*> SASA::create_sasa(int sa_channel_rows,int sa_channel_columns, int sa_channels){
	
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



std::vector<fMat> SASA::input_tensor_transformer(std::vector<fMat> input_tensor,std::vector<ConvTransformer*> CT_ptr){

	std::vector<fMat> transformed_mats(input_tensor.size());
	for(int i = 0 ; i < input_tensor.size() ; i ++){
		transformed_mats.at(i).push_back( CT_ptr.at(i)->(fMat)transform(mat2v<float,float>(input_tensor.at(i), input_tensor.at(i).size(),input_tensor.at(i).at(i).size()))); 
	}
	return transformed_mats;

}

std::vector<float> SASA:: load_kernel_tensors(std::vector<std::vector<fMat>> input_kernel, int kernel_channel,int kernel_number){



	std::vector<float> kernel_tensor(sa_channel_rows*sa_channel_columns);
	std::vector<float> output_tensor(sa_channel_rows*sa_channel_columns);


	for(int i = kernel_number ; i < (kernel_number+ sa_channel_columns) ; i ++){
		
		kernel_tensor = mat2v<float,float>(input_kernel.at(i).at(kernel_channel), input_kernel.at(0).at(0).size(),input_kernel.at(0).at(0).at(0).size());

		for(int j = 0 ; j < sa_channel_rows; j++){
		
			output_tensor.push_back(kernel_tensor[j]);
		}
		
	}

	return output_tensor;
	
}


void SASA::load_weights_tensor(SA* SA_ptr, ConvTransformer* CT_ptr, std::vector<float> input) {
    // transform weights is a func of Convtransform
	SA_ptr->load_weights(CT_ptr->transform_weights(input,sa_channel_rows,sa_channel_columns));
	
}


fMat SASA::operatorr(fMat transformed_mats, SA* SA_ptr, ConvTransformer* CT_ptr){

	std::vector<fMat> adder_mat(8); 
	std::vector<fMat> output;
	std::vector<float> vec;
	fMat out_mat;
	
	for(int i = 0 ; i < sa_channels; i ++){
			// yaha pr b wo dekhna padega jo limited jaga h fpga pr aur phir reuse krte h
		
			SA_ptr->propagate(transformed_mats);
			output = SA_ptr->get_output(); 
			vec = CT_ptr->untransform(output);

			out_mat = v2mat<float,float>(vec,sa_channel_columns,vec.size()/sa_channel_columns);

			// here it is filling the vec (linear) with all the values upto kernel 7 (sa_column)	
	}
	return out_mat;
}

/* the input is in the form of : channel-> kernel -> elements  (e.g. C0 -> K0 - > C0K0[elements])
   the output will have the output stored in only one channel(0) having the total number of kernels ... 
   that channel will have no significance of its index .
*/

fMat SASA:: adder(std::vector<fMat> input){


	for(int m = 0 ; m < input.at(0).size(); m ++){
			for(int n = 0 ; n < input.size(); n ++){ 
				for(int p = 0 ; p < input.at(0).at(0).size(); p ++){
					input.at(0).at(m).at(p)= input.at(0 ).at(m ).at(p) + input.at(n+1).at(m).at(p);
				}
			return input.at(0);
			}
		}
}

void SASA::splitter(std::vector<std::vector<fMat>>&vec , fMat temp_mat , int channel_number , int kernel_number){
	for(int i = kernel_number ; i < (kernel_number + temp_mat.size()) ; i ++){
		vec.at(channel_number).at(i).push_back( temp_mat.at(i));
	}
}

void SASA::master(std::vector<fMat> input_tensor,std::vector<std::vector<fMat>> input_kernel){  //NCHW         // master/ Control Unit
	
	std::vector<SA*> SA_ptr = create_sasa(sa_channel_rows,sa_channel_columns,sa_channels);
	std::vector<fMat> transformed_mats;
	std::vector<float> output_weights;
	std::vector<std::vector<fMat>> vec;   // channel pointers -> kernel pointers (0-7) -> linear upto c0k7
	fMat temp_mat;
	fMat output_mat;
	std::vector<ConvTransformer*> CT_ptr= create_ConvTransformer(input_tensor.at(0).size(),input_tensor.at(0).at(0).size(),
																input_kernel.at(0).at(0).size(),input_kernel.at(0).at(0).at(0).size(),
																sa_channel_rows,sa_channel_columns);

	transformed_mats = input_tensor_transformer(input_tensor, CT_ptr);

/* master is filling the SAs kernel wise ... filling up to its capacity and then reloading the
*  SAs with same channels but different kernel and then after all the kernels are done ....
* channel is reloaded and then the process is done all over again.
*/

	for(int k = 0 ;  k < input_kernel.at(0).size(); k ++ ){
		for( int i = 0 ; i < input_kernel.size(); i ++){
			for( int j = 0 ; j < sa_channels; j++){
		
				output_weights= load_kernel_tensors(input_kernel, (k*sa_channels+j) , i*8);
				load_weights_tensor(SA_ptr.at(j), CT_ptr.at(j), output_weights);
				// vec.at(i).at(i*sa_channel_columns + j).push_back(operatorr(transformed_mats.at(j),SA_ptr.at(j), CT_ptr.at(j)));
				temp_mat = operatorr(transformed_mats.at(j),SA_ptr.at(j), CT_ptr.at(j));
				splitter(vec,temp_mat, (k*sa_channels+j), i*8 );

			}
		}	
	}
	output_mat = adder(vec);
}


/* TODO:
check the ' at() ' of all vectors , if they exist or not, before 'push_back'ing into them
templates for the demanding functions 
*/

