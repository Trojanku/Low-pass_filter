#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime.h>

#include <sstream>

using namespace std;

#include "../EasyBMP.h"
#include "../EasyBMP.cpp"

__global__ void DFT_GPU_series(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *high, int *width, int last_x, int last_y) {

	int dir = 1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx;
	int idy;

	int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	idx = id_x + last_x;
	idy = id_y + last_y;

	if (idx < *width && idy < *high)
	{
		arg = -dir * 2.0 * 3.141592654 * (double)(idx - *width / 2) / (double)*width;

		Re_2[idy**width + idx] = 0;
		Im_2[idy**width + idx] = 0;

		for (k = -*width / 2; k < *width / 2;k++) {

			cosarg = cos(k * arg);
			sinarg = sin(k * arg);

			Re_2[idy**width + idx] += (Re_1[idy**width + k + *width / 2] * cosarg - Im_1[idy**width + k + *width / 2] * sinarg);
			Im_2[idy**width + idx] += (Re_1[idy**width + k + *width / 2] * sinarg + Im_1[idy**width + k + *width / 2] * cosarg);
		}
	}
}

__global__ void DFT_GPU_columns(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *high, int *width, int last_x, int last_y) {

	int dir = 1;
	long k;
	double arg;
	double cosarg, sinarg;

	int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	int idx = 0;
	int idy = 0;

	idx = id_x + last_x;
	idy = id_y + last_y;

	arg = -dir * 2.0 * 3.141592654 * (double)(idy - *high / 2) / (double)*high;

	if (idx < *width && idy < *high)
	{
		Re_2[idy**width + idx] = 0;
		Im_2[idy**width + idx] = 0;


		for (k = -*high / 2; k < *high / 2;k++) {


			cosarg = cos(k * arg);
			sinarg = sin(k * arg);

			Re_2[idy**width + idx] += (Re_1[(k + *high / 2)**width + idx] * cosarg - Im_1[(k + *high / 2) **width + idx] * sinarg);
			Im_2[idy**width + idx] += (Re_1[(k + *high / 2)**width + idx] * sinarg + Im_1[(k + *high / 2)**width + idx] * cosarg);
		}
	}
}

__global__ void IDFT_GPU_series(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *high, int *width, int last_x, int last_y) {


	int dir = -1;
	long k;
	double arg;
	double cosarg, sinarg;

	int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	int idx = id_x + last_x;
	int idy = id_y + last_y;


	arg = -dir * 2.0 * 3.141592654 * (double)(idx - *width / 2) / (double)*width;

	if (idx < *width && idy < *high)
	{
		Re_2[idy**width + idx] = 0;
		Im_2[idy**width + idx] = 0;


		for (k = -*width / 2; k < *width / 2;k++)
		{

			cosarg = cos(k * arg);
			sinarg = sin(k * arg);

			Re_2[idy**width + idx] += (Re_1[idy**width + k + *width / 2] * cosarg - Im_1[idy**width + k + *width / 2] * sinarg);
			Im_2[idy**width + idx] += (Re_1[idy**width + k + *width / 2] * sinarg + Im_1[idy**width + k + *width / 2] * cosarg);

		}

		Re_2[idy**width + idx] = Re_2[idy**width + idx] / (*high);
		Im_2[idy**width + idx] = Im_2[idy**width + idx] / (*high);
	}

}

__global__ void IDFT_GPU_columns(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *high, int *width, int last_x, int last_y) {


	int dir = -1;
	long k;
	double arg;
	double cosarg, sinarg;

	int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	int idx = id_x + last_x;
	int idy = id_y + last_y;


	arg = -dir * 2.0 * 3.141592654 * (double)(idy - *high / 2) / (double)*high;

	if (idx < *width && idy < *high)
	{
		Re_2[idy**width + idx] = 0;
		Im_2[idy**width + idx] = 0;


		for (k = -*high / 2;k < *high / 2;k++)
		{

			cosarg = cos(k * arg);
			sinarg = sin(k * arg);


			Re_2[idy**width + idx] += (Re_1[(k + *high / 2)**width + idx] * cosarg - Im_1[(k + *high / 2) **width + idx] * sinarg);
			Im_2[idy**width + idx] += (Re_1[(k + *high / 2)**width + idx] * sinarg + Im_1[(k + *high / 2)**width + idx] * cosarg);
		}


		Re_2[idy**width + idx] = Re_2[idy**width + idx] / (*width);
		Im_2[idy**width + idx] = Im_2[idy**width + idx] / (*width);
	}
}

void cross(double *Re, double *Im, int high, int width,int R)
{
	int Dy = (high / 2 - R);
	int Dx = (width / 2 - R);

	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			if (sqrt(pow(((high / 2) - j) / ((high / 2) - Dy), 2) + pow(((width / 2) - i) / ((width / 2) - Dx), 2)) > 1)
			{
				Re[j*width + i] = 0;
				Im[j*width + i] = 0;
			}

		}
	}
}

void circle(double *Re, double *Im, int high, int width, int R)
{
	int center_x = int(width / 2);
	int center_y = int(high / 2);

	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			if( pow(i - center_x, 2) + pow(j - center_y, 2) > pow(R, 2))
			{
				Re[j*width + i] = 0;
				Im[j*width + i] = 0;
			}
		}
	}
}


void maska(double *Re, double *Im, int high, int width, int R)
{
	circle(Re,Im,high,width, R);

}

void zapisanie_obrazu(BMP output, char file [])
{
	output.WriteToFile(file);
}


int main(int argc, char** argv)
{
	// Default values

	int R = 30;  //  R for mask	
	int threads = 32;
	char  input [] = "input.bmp";
	char output [] = "out.bmp";
	int grid_x = 16;
	int grid_y = 16;
	int block_x = 32;
	int block_y = 32;

	for (int i = 1; i < argc ; i ++){
		std::stringstream ss(argv[i]);
		if(ss.str() == "-f"){
			ss.str(argv[i + 1]);
			if(!(ss >> input)){
				cout << "Failed to load input file" << endl;
			}	
		}
		else if(ss.str() == "-t"){
			ss.str(argv[i + 1]);
			if(!(ss >> threads)){
				cout << "Failed to set threads variable" << endl;
			}
		}
		else if(ss.str() == "-r"){
			ss.str(argv[i + 1]);
			if(!(ss >> R)){
				cout << "Failed to set R variable" << endl;
			}
		}
		else if(ss.str() == "-grid"){
			ss.str(argv[i + 1]);
			if(!(ss >> grid_x)){
				cout << "Failed to set grid_x variable" << endl;
			}
			std::stringstream ss(argv[i + 2]);
			if(!(ss >> grid_y)){
				cout << "Failed to set grid_y variable" << endl;
			}
		}
		else if(ss.str() == "-block"){
			ss.str(argv[i + 1]);
			if(!(ss >> block_x)){
				cout << "Failed to set thread_x variable" << endl;
			}
			std::stringstream ss(argv[i + 2]);
			if(!(ss >> block_y)){
				cout << "Failed to set thread_y variable" << endl;
			}
		}
	}

	cout << "Mask size R = " <<  R << endl;	
	cout << "Grid size = (" <<  grid_x << " , " << grid_y << ")" << endl;	
	cout << "Block size = (" <<  block_x << " , " << block_y << ")" <<  endl;	

	BMP Input;
	if(!(Input.ReadFromFile(input)));
		cout << "Input file: " << input << " succesfully loaded" << endl;
			
	cout << "Threads number: " << threads << endl;
	int size = Input.TellHeight()*Input.TellWidth();
	cout << "Total size: " << size << endl;

	int width = Input.TellWidth();
	int high = Input.TellHeight();

	double *Pixels_Re = new double[size];
	double *Pixels_Im = new double[size];

	int z = 0;


	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			int Temp = (int)floor(0.299*Input(i, j)->Red +
				0.587*Input(i, j)->Green +
				0.114*Input(i, j)->Blue);


			Pixels_Re[z] = Temp;
			Pixels_Im[z] = 0;
			z++;

			Input(i, j)->Red = 0;
			Input(i, j)->Green = 0;
			Input(i, j)->Blue = 0;
		}
	}
	
	// // //                                          !!!!!         GPU          !!!!!!

	// variables init

	double *Pixels_Re_d = new double[size];
	double *Pixels_Im_d = new double[size];

	double *Re_2_d = new double[size];
	double *Im_2_d = new double[size];
	double *Re_3_d = new double[size];
	double *Im_3_d = new double[size];
	double *Re_4_d = new double[size];
	double *Im_4_d = new double[size];

	double *Re_5_d = new double[size];
	double *Im_5_d = new double[size];

	int * width_d;
	int * high_d;

	int last_x;   // this variable helps when there is less threads then pixels ( kernel is run more times then one)
	int last_y;

	// Allocation gpu memory


	cudaMalloc((void**)&Pixels_Re_d, sizeof(double)*size);   
	cudaMalloc((void**)&Pixels_Im_d, sizeof(double)*size);
	cudaMalloc((void**)&Im_2_d, sizeof(double)*size);
	cudaMalloc((void**)&Re_2_d, sizeof(double)*size);
	cudaMalloc((void**)&Im_3_d, sizeof(double)*size);
	cudaMalloc((void**)&Re_3_d, sizeof(double)*size);
	cudaMalloc((void**)&Im_4_d, sizeof(double)*size);
	cudaMalloc((void**)&Re_4_d, sizeof(double)*size);
	cudaMalloc((void**)&Im_5_d, sizeof(double)*size);
	cudaMalloc((void**)&Re_5_d, sizeof(double)*size);
	cudaMalloc((void**)&high_d, sizeof(int));
	cudaMalloc((void**)&width_d, sizeof(int));	

	// Rewrite data from CPU to GPU

	cout << "Copy from CPU to GPU...";
	cudaMemcpy(Pixels_Re_d, Pixels_Re, sizeof(double)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(Pixels_Im_d, Pixels_Im, sizeof(double)*size, cudaMemcpyHostToDevice);

	cudaMemcpy(high_d, &high, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);

	cout << " done" << endl;


	//dim3 grid_dim((width + (threads - 1)) / threads, (high + (threads - 1)) / threads, 1);
	//dim3 block_dim(threads, threads);

	dim3 grid_dim(grid_x, grid_y, 1);
	dim3 block_dim(block_x, block_y);

    //  GPU transform

	// Compute DFT, split into smaller tasks if number of threads is less then pnumber of pixels
	cout << "DFT processing...";
	while(last_x < width)
	{
		last_y = 0;
		while(last_y < high)
		{
			DFT_GPU_series << < grid_dim, block_dim >> >(Pixels_Re_d, Pixels_Im_d, Im_2_d, Re_2_d, high_d, width_d, last_x, last_y);
			last_y += block_dim.y * grid_dim.y;
		}
		last_x += block_dim.x * grid_dim.x;
	}
	cudaDeviceSynchronize();
	last_x = 0;
	while(last_x < width)
	{
		last_y = 0;
		while(last_y < high)
		{
			DFT_GPU_columns << < grid_dim, block_dim >> >(Re_2_d, Im_2_d, Im_3_d, Re_3_d, high_d, width_d, last_x, last_y);	
			last_y += block_dim.y * grid_dim.y;
		}
		last_x += block_dim.x * grid_dim.x;
	}

	cudaMemcpy(Pixels_Re, Re_3_d, sizeof(double)*width*high, cudaMemcpyDeviceToHost);
	cudaMemcpy(Pixels_Im, Im_3_d, sizeof(double)*width*high, cudaMemcpyDeviceToHost);

	cout << " done" << endl;	


	cout << "Masking...";	
	maska(Pixels_Re, Pixels_Im, high, width, R);

	cudaMemcpy(Pixels_Re_d, Pixels_Re, sizeof(double)*high*width, cudaMemcpyHostToDevice);
	cudaMemcpy(Pixels_Im_d, Pixels_Im, sizeof(double)*high*width, cudaMemcpyHostToDevice);

	cout << " done" << endl;	

	cout << "IDFT processing...";

	// Compute IDFT

	last_x = 0;
	last_y = 0;

	while(last_x < width)
	{
		last_y = 0;
		while(last_y < high)
		{
			IDFT_GPU_series << < grid_dim, block_dim >> >(Pixels_Re_d, Pixels_Im_d, Im_4_d, Re_4_d, high_d, width_d,last_x, last_y);
			last_y += block_dim.y * grid_dim.y;
		}
		last_x += block_dim.x * grid_dim.x;
	}
	cudaDeviceSynchronize();
	last_x = 0;
	while(last_x < width)
	{
		last_y = 0;
		while(last_y < high)
		{	
			IDFT_GPU_columns << < grid_dim, block_dim >> >(Re_4_d, Im_4_d, Im_5_d, Re_5_d, high_d, width_d, last_x, last_y);
			last_y += block_dim.y * grid_dim.y;
		}
		last_x += block_dim.x * grid_dim.x;
	}

	// Copy data back to host

	cudaMemcpy(Pixels_Re, Re_5_d, sizeof(double)*width*high, cudaMemcpyDeviceToHost);
	cudaMemcpy(Pixels_Im, Im_5_d, sizeof(double)*width*high, cudaMemcpyDeviceToHost);

	cout << " done" << endl;	

	int g = 0;
	float temp = 0;

	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			temp = sqrt(pow(Pixels_Re[g], 2) + pow(Pixels_Im[g], 2));
			Input(i, j)->Red = temp;
			Input(i, j)->Green = temp;
			Input(i, j)->Blue = temp;
			g++;
		}
	}


	zapisanie_obrazu(Input, output);
	cout << "Saved to: " << output <<  endl;	

	// clean up

	delete[] Pixels_Re;
	delete[] Pixels_Im;


	cudaFree(Pixels_Re_d);
	cudaFree(Pixels_Im_d);
	cudaFree(Re_2_d);
	cudaFree(Im_2_d);
	cudaFree(Re_3_d);
	cudaFree(Im_3_d);
	cudaFree(Re_4_d);
	cudaFree(Im_4_d);
	cudaFree(Re_5_d);
	cudaFree(Im_5_d);
	cudaFree(width_d);
	cudaFree(high_d);


	return 0;
}
