#include <iostream>
#include "sim.h"
#include "transformers.h"
#include "utils.h"

class SASA{
	private:

		int sa_channels;
		int sa_channel_rows;
		int sa_channel_columns;

	public:

		SASA(int sa_channel_rows,int sa_channel_columns,int sa_channels);
		// ~SASA();
		std::vector<SA*> create_sasa(int sa_channel_rows,int sa_channel_columns, int sa_channels);
		std::vector<ConvTransformer*> create_ConvTransformer(int IW, int IH, int KW, int KH, int srows, int scols);

		std::vector<Mat> input_tensor_transformer(std::vector<Mat> input_tensor, std::vector<ConvTransformer*> CT_ptr);  
		std::vector<int> load_kernel_tensors(std::vector<std::vector<Mat>> input_kernel, int kernel_channel, int kernel_number); // NCHW
		void load_weights_tensor(SA* SA_ptr, ConvTransformer* CT_ptr, std::vector<int> input);
		void master(std::vector<Mat> input_tensor,std::vector<std::vector<Mat>> input_kernel);
		Mat operatorr(Mat transformed_mats, SA* SA_ptr, ConvTransformer* CT_ptr);
		void splitter(std::vector<Mat>&vec , Mat temp_mat , int channel_number, int kernel_number,int input_kernel_size);
		Mat adder(std::vector<Mat> input);
	
};