#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cuda.h>
#include "device_launch_parameters.h"
#include<device_functions.h>
#include <cuda_runtime.h>

using namespace std;

#include "EasyBMP.h"
#include "EasyBMP.cpp"

int const size = 16;


__global__ void DFT_GPU_series(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *high, int *width) {

	int dir = 1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	arg = -dir * 2.0 * 3.141592654 * (double)(idx - *width / 2) / (double)*width;

	if (idx < *width && idy < *high)
	{
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

__global__ void DFT_GPU_columns(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *high, int *width) {

	int dir = 1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;


	arg = -dir * 2.0 * 3.141592654 * (double)(idy - *high / 2) / (double)*high;

	if (idx < *width && idy < *high)
	{
		Re_2[idy**width + idx] = 0;
		Im_2[idy**width + idx] = 0;


		for (k = -*high / 2;k < *high / 2;k++) {


			cosarg = cos(k * arg);
			sinarg = sin(k * arg);

			Re_2[idy**width + idx] += (Re_1[(k + *high / 2)**width + idx] * cosarg - Im_1[(k + *high / 2) **width + idx] * sinarg);
			Im_2[idy**width + idx] += (Re_1[(k + *high / 2)**width + idx] * sinarg + Im_1[(k + *high / 2)**width + idx] * cosarg);
		}
	}

}

__global__ void IDFT_GPU_series(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *high, int *width) {


	int dir = -1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;


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

__global__ void IDFT_GPU_columns(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *high, int *width) {


	int dir = -1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;


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



void maska(double *Re, double *Im, int high, int width)
{

	int R = 100;
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

int DFT(int size, double *Re_1, double *Im_1, int high, int width)
{
	int dir = 1;
	long i, k;
	double arg;
	double cosarg, sinarg;
	double *Re_2 = new double[size];
	double *Im_2 = new double[size];



	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			arg = -dir * 2.0 * 3.141592654 * (double)i / (double)width;
			Re_2[j*width + i] = 0;
			Im_2[j*width + i] = 0;

			for (k = 0;k < width;k++) {

				cosarg = cos(k * arg);
				sinarg = sin(k * arg);

				Re_2[j*width + i] += (Re_1[j*width + k] * cosarg - Im_1[j*width + k] * sinarg);
				Im_2[j*width + i] += (Re_1[j*width + k] * sinarg + Im_1[j*width + k] * cosarg);
			}
		}
	}

	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			arg = -dir * 2.0 * 3.141592654 * (double)i / (double)width;
			for (k = 0;k < high; k++) {

				cosarg = cos(k * arg);
				sinarg = sin(k * arg);

				Re_2[i*high + j] += (Re_1[k*high + j] * cosarg - Im_1[k*high + j] * sinarg);
				Im_2[i*high + j] += (Re_1[k*high + j] * sinarg + Im_1[k*high + j] * cosarg);
			}
		}
	}


	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			Re_1[j*width + i] = Re_2[j*width + i];
			Im_1[j*width + i] = Im_2[j*width + i];
		}
	}


	free(Re_2);
	free(Im_2);

	return(1);
}

int IDFT(int size, double *Re_1, double *Im_1, int high, int width)
{
	int dir = -1;
	long i, k;
	double arg;
	double cosarg, sinarg;
	double *Re_2 = new double[size];
	double *Im_2 = new double[size];

	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			arg = -dir * 2.0 * 3.141592654 * (double)i / (double)width;
			Re_2[j*width + i] = 0;
			Im_2[j*width + i] = 0;

			for (k = 0;k < width;k++) {

				cosarg = cos(k * arg);
				sinarg = sin(k * arg);

				Re_2[j*width + i] += (Re_1[j*width + k] * cosarg - Im_1[j*width + k] * sinarg);
				Im_2[j*width + i] += (Re_1[j*width + k] * sinarg + Im_1[j*width + k] * cosarg);
			}

		}
	}

	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			arg = -dir * 2.0 * 3.141592654 * (double)i / (double)width;
			for (k = 0;k < high; k++) {

				cosarg = cos(k * arg);
				sinarg = sin(k * arg);

				Re_2[i*high + j] += (Re_1[k*high + j] * cosarg - Im_1[k*high + j] * sinarg);
				Im_2[i*high + j] += (Re_1[k*high + j] * sinarg + Im_1[k*high + j] * cosarg);
			}
		}
	}


	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			Re_1[j*width + i] = Re_2[j*width + i] / (2 * width);
			Im_1[j*width + i] = Im_2[j*width + i] / (2 * width);

		}
	}


	free(Re_2);
	free(Im_2);

	return(1);
}



BMP wczytywanie_obrazu()
{
	BMP Input;

	Input.ReadFromFile("1920x1080.bmp");

	return Input;


}

void zapisanie_obrazu(BMP output)
{
	output.WriteToFile("wyjscie.bmp");

}


int main()
{
	BMP Input = wczytywanie_obrazu();

	int size = Input.TellHeight()*Input.TellWidth();
	cout << size;

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

	//DFT(size, Pixels_Re,Pixels_Im,high,width);

	//maska(Pixels_Re, Pixels_Im, high, width);

	//IDFT(size, Pixels_Re, Pixels_Im, high, width);



	//                                          !!!!!         GPU          !!!!!!

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

	cudaMemcpy(Pixels_Re_d, Pixels_Re, sizeof(double)*size, cudaMemcpyHostToDevice);
	cudaMemcpy(Pixels_Im_d, Pixels_Im, sizeof(double)*size, cudaMemcpyHostToDevice);

	cudaMemcpy(high_d, &high, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);


	
	dim3 grid((width + 31) / 32, (high + 31) / 32, 1);
	dim3 threads(32, 32);

    //  GPU transform

	DFT_GPU_series << < grid, threads >> >(Pixels_Re_d, Pixels_Im_d, Im_2_d, Re_2_d, high_d, width_d);

	DFT_GPU_columns << < grid, threads >> >(Re_2_d, Im_2_d, Im_3_d, Re_3_d, high_d, width_d);


	cudaMemcpy(Pixels_Re, Re_3_d, sizeof(double)*width*high, cudaMemcpyDeviceToHost);
	cudaMemcpy(Pixels_Im, Im_3_d, sizeof(double)*width*high, cudaMemcpyDeviceToHost);

    
	maska(Pixels_Re, Pixels_Im, high, width);

	cudaMemcpy(Pixels_Re_d, Pixels_Re, sizeof(double)*high*width, cudaMemcpyHostToDevice);
	cudaMemcpy(Pixels_Im_d, Pixels_Im, sizeof(double)*high*width, cudaMemcpyHostToDevice);

	IDFT_GPU_series << < grid, threads >> >(Pixels_Re_d, Pixels_Im_d, Im_4_d, Re_4_d, high_d, width_d);

	IDFT_GPU_columns << < grid, threads >> >(Re_4_d, Im_4_d, Im_5_d, Re_5_d, high_d, width_d);

	cudaMemcpy(Pixels_Re, Re_5_d, sizeof(double)*width*high, cudaMemcpyDeviceToHost);
	cudaMemcpy(Pixels_Im, Im_5_d, sizeof(double)*width*high, cudaMemcpyDeviceToHost);


	int g = 0;
	for (int j = 0; j < high; j++)
	{
		for (int i = 0; i < width; i++)
		{
			Input(i, j)->Red = sqrt(pow(Pixels_Re[g], 2) + pow(Pixels_Im[g], 2));
			Input(i, j)->Green = sqrt(pow(Pixels_Re[g], 2) + pow(Pixels_Im[g], 2));
			Input(i, j)->Blue = sqrt(pow(Pixels_Re[g], 2) + pow(Pixels_Im[g], 2));
			g++;
		}
	}


	zapisanie_obrazu(Input);


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