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


__global__ void DFT_GPU_wiersze(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *wysokosc, int *szerokosc) {

	int dir = 1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	// po wierszach i kolumnach

	arg = -dir * 2.0 * 3.141592654 * (double)(idx - *szerokosc / 2) / (double)*szerokosc;

	if (idx < *szerokosc && idy < *wysokosc)
	{
		Re_2[idy**szerokosc + idx] = 0;
		Im_2[idy**szerokosc + idx] = 0;


		for (k = -*szerokosc / 2; k < *szerokosc / 2;k++) {

			cosarg = cos(k * arg);
			sinarg = sin(k * arg);

			Re_2[idy**szerokosc + idx] += (Re_1[idy**szerokosc + k + *szerokosc / 2] * cosarg - Im_1[idy**szerokosc + k + *szerokosc / 2] * sinarg);
			Im_2[idy**szerokosc + idx] += (Re_1[idy**szerokosc + k + *szerokosc / 2] * sinarg + Im_1[idy**szerokosc + k + *szerokosc / 2] * cosarg);
		}

	}
}

__global__ void DFT_GPU_kolumny(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *wysokosc, int *szerokosc) {

	int dir = 1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	// po wierszach i kolumnach

	arg = -dir * 2.0 * 3.141592654 * (double)(idy - *wysokosc / 2) / (double)*wysokosc;

	if (idx < *szerokosc && idy < *wysokosc)
	{
		Re_2[idy**szerokosc + idx] = 0;
		Im_2[idy**szerokosc + idx] = 0;


		for (k = -*wysokosc / 2;k < *wysokosc / 2;k++) {


			cosarg = cos(k * arg);
			sinarg = sin(k * arg);

			Re_2[idy**szerokosc + idx] += (Re_1[(k + *wysokosc / 2)**szerokosc + idx] * cosarg - Im_1[(k + *wysokosc / 2) **szerokosc + idx] * sinarg);
			Im_2[idy**szerokosc + idx] += (Re_1[(k + *wysokosc / 2)**szerokosc + idx] * sinarg + Im_1[(k + *wysokosc / 2)**szerokosc + idx] * cosarg);
		}
	}

}

__global__ void IDFT_GPU_wiersze(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *wysokosc, int *szerokosc) {


	int dir = -1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	// po wierszach i kolumnach

	arg = -dir * 2.0 * 3.141592654 * (double)(idx - *szerokosc / 2) / (double)*szerokosc;

	if (idx < *szerokosc && idy < *wysokosc)
	{
		Re_2[idy**szerokosc + idx] = 0;
		Im_2[idy**szerokosc + idx] = 0;


		for (k = -*szerokosc / 2; k < *szerokosc / 2;k++)
		{

			cosarg = cos(k * arg);
			sinarg = sin(k * arg);

			Re_2[idy**szerokosc + idx] += (Re_1[idy**szerokosc + k + *szerokosc / 2] * cosarg - Im_1[idy**szerokosc + k + *szerokosc / 2] * sinarg);
			Im_2[idy**szerokosc + idx] += (Re_1[idy**szerokosc + k + *szerokosc / 2] * sinarg + Im_1[idy**szerokosc + k + *szerokosc / 2] * cosarg);

		}

		Re_2[idy**szerokosc + idx] = Re_2[idy**szerokosc + idx] / (*wysokosc);
		Im_2[idy**szerokosc + idx] = Im_2[idy**szerokosc + idx] / (*wysokosc);
		Im_2[idy**width + idx] = Im_2[idy**width + idx] / (*high);
	}

}

__global__ void IDFT_GPU_kolumny(double *Re_1, double *Im_1, double *Im_2, double *Re_2, int *wysokosc, int *szerokosc) {


	int dir = -1;
	long k;
	double arg;
	double cosarg, sinarg;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	// po wierszach i kolumnach

	arg = -dir * 2.0 * 3.141592654 * (double)(idy - *wysokosc / 2) / (double)*wysokosc;

	if (idx < *szerokosc && idy < *wysokosc)
	{
		Re_2[idy**szerokosc + idx] = 0;
		Im_2[idy**szerokosc + idx] = 0;


		for (k = -*wysokosc / 2;k < *wysokosc / 2;k++)
		{

			cosarg = cos(k * arg);
			sinarg = sin(k * arg);


			Re_2[idy**szerokosc + idx] += (Re_1[(k + *wysokosc / 2)**szerokosc + idx] * cosarg - Im_1[(k + *wysokosc / 2) **szerokosc + idx] * sinarg);
			Im_2[idy**szerokosc + idx] += (Re_1[(k + *wysokosc / 2)**szerokosc + idx] * sinarg + Im_1[(k + *wysokosc / 2)**szerokosc + idx] * cosarg);
		}


		Re_2[idy**szerokosc + idx] = Re_2[idy**szerokosc + idx] / (*szerokosc);
		Im_2[idy**szerokosc + idx] = Im_2[idy**szerokosc + idx] / (*szerokosc);
	}
}



void maska(double *Re, double *Im, int wysokosc, int szerokosc)
{

	//if (j  <  wysokosc / 2 - d / 2 || j  >  wysokosc / 2 + d / 2 || i  <  szerokosc / 2 - d / 2 || i  > szerokosc / 2 + d / 2)
	//if ((j  >  wysokosc / 2 - d / 2) && (j  <  wysokosc / 2 + d / 2) && (i  >  szerokosc / 2 - d / 2) && (i  < szerokosc / 2 + d / 2))
	//if ((j  >  d) &&( j  <  wysokosc -d ) || ( i  > d )&&( i  < szerokosc - d))

	//int d = szerokosc/100 * 20 ;

	int R = 100;
	int Dy = (wysokosc / 2 - R);
	int Dx = (szerokosc / 2 - R);

	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{

			if (sqrt(pow(((wysokosc / 2) - j) / ((wysokosc / 2) - Dy), 2) + pow(((szerokosc / 2) - i) / ((szerokosc / 2) - Dx), 2)) > 1)
			{
				Re[j*szerokosc + i] = 0;
				Im[j*szerokosc + i] = 0;
			}

		}
	}
}

/*
Direct fourier transform
*/
int DFT(int rozmiar, double *Re_1, double *Im_1, int wysokosc, int szerokosc)
{
	int dir = 1;
	long i, k;
	double arg;
	double cosarg, sinarg;
	double *Re_2 = new double[rozmiar];
	double *Im_2 = new double[rozmiar];



	// po wierszach
	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{
			arg = -dir * 2.0 * 3.141592654 * (double)i / (double)szerokosc;
			Re_2[j*szerokosc + i] = 0;
			Im_2[j*szerokosc + i] = 0;

			for (k = 0;k < szerokosc;k++) {

				cosarg = cos(k * arg);
				sinarg = sin(k * arg);

				Re_2[j*szerokosc + i] += (Re_1[j*szerokosc + k] * cosarg - Im_1[j*szerokosc + k] * sinarg);
				Im_2[j*szerokosc + i] += (Re_1[j*szerokosc + k] * sinarg + Im_1[j*szerokosc + k] * cosarg);
			}
		}
	}


	// po kolumnach
	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{
			arg = -dir * 2.0 * 3.141592654 * (double)i / (double)szerokosc;
			for (k = 0;k < wysokosc; k++) {

				cosarg = cos(k * arg);
				sinarg = sin(k * arg);

				Re_2[i*wysokosc + j] += (Re_1[k*wysokosc + j] * cosarg - Im_1[k*wysokosc + j] * sinarg);
				Im_2[i*wysokosc + j] += (Re_1[k*wysokosc + j] * sinarg + Im_1[k*wysokosc + j] * cosarg);
			}
		}
	}

	/* Copy the data back */

	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{
			Re_1[j*szerokosc + i] = Re_2[j*szerokosc + i];
			Im_1[j*szerokosc + i] = Im_2[j*szerokosc + i];
		}
	}


	free(Re_2);
	free(Im_2);

	return(1);
}

int IDFT(int rozmiar, double *Re_1, double *Im_1, int wysokosc, int szerokosc)
{
	int dir = -1;
	long i, k;
	double arg;
	double cosarg, sinarg;
	double *Re_2 = new double[rozmiar];
	double *Im_2 = new double[rozmiar];

	// po wierszach
	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{
			arg = -dir * 2.0 * 3.141592654 * (double)i / (double)szerokosc;
			Re_2[j*szerokosc + i] = 0;
			Im_2[j*szerokosc + i] = 0;

			for (k = 0;k < szerokosc;k++) {

				cosarg = cos(k * arg);
				sinarg = sin(k * arg);

				Re_2[j*szerokosc + i] += (Re_1[j*szerokosc + k] * cosarg - Im_1[j*szerokosc + k] * sinarg);
				Im_2[j*szerokosc + i] += (Re_1[j*szerokosc + k] * sinarg + Im_1[j*szerokosc + k] * cosarg);
			}

		}
	}

	/* Copy the data back */

	// po kolumnach
	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{
			arg = -dir * 2.0 * 3.141592654 * (double)i / (double)szerokosc;
			for (k = 0;k < wysokosc; k++) {

				cosarg = cos(k * arg);
				sinarg = sin(k * arg);

				Re_2[i*wysokosc + j] += (Re_1[k*wysokosc + j] * cosarg - Im_1[k*wysokosc + j] * sinarg);
				Im_2[i*wysokosc + j] += (Re_1[k*wysokosc + j] * sinarg + Im_1[k*wysokosc + j] * cosarg);
			}
		}
	}

	/* Copy the data back */

	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{
			Re_1[j*szerokosc + i] = Re_2[j*szerokosc + i] / (2 * szerokosc);
			Im_1[j*szerokosc + i] = Im_2[j*szerokosc + i] / (2 * szerokosc);

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

	int rozmiar = Input.TellHeight()*Input.TellWidth();
	cout << rozmiar;

	int szerokosc = Input.TellWidth();
	int wysokosc = Input.TellHeight();

	double *piksele_Re = new double[rozmiar];
	double *piksele_Im = new double[rozmiar];

	int z = 0;


	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{
			int Temp = (int)floor(0.299*Input(i, j)->Red +
				0.587*Input(i, j)->Green +
				0.114*Input(i, j)->Blue);


			piksele_Re[z] = Temp;
			piksele_Im[z] = 0;
			z++;

			Input(i, j)->Red = 0;
			Input(i, j)->Green = 0;
			Input(i, j)->Blue = 0;


		}
	}

	//DFT(rozmiar, piksele_Re,piksele_Im,wysokosc,szerokosc);

	//maska(piksele_Re, piksele_Im, wysokosc, szerokosc);

	//IDFT(rozmiar, piksele_Re, piksele_Im, wysokosc, szerokosc);



	//                                          !!!!!         GPU          !!!!!!

	// zdefiniowanie zmiennych

	double *piksele_Re_d = new double[rozmiar];
	double *piksele_Im_d = new double[rozmiar];

	double *Re_2_d = new double[rozmiar];
	double *Im_2_d = new double[rozmiar];
	double *Re_3_d = new double[rozmiar];
	double *Im_3_d = new double[rozmiar];
	double *Re_4_d = new double[rozmiar];
	double *Im_4_d = new double[rozmiar];

	double *Re_5_d = new double[rozmiar];
	double *Im_5_d = new double[rozmiar];

	int * szerokosc_d;
	int * wysokosc_d;

	// 1. Alokowanie pamieci GPU


	cudaMalloc((void**)&piksele_Re_d, sizeof(double)*rozmiar);   //alokowanie pamieci GPU
	cudaMalloc((void**)&piksele_Im_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&Im_2_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&Re_2_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&Im_3_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&Re_3_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&Im_4_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&Re_4_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&Im_5_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&Re_5_d, sizeof(double)*rozmiar);
	cudaMalloc((void**)&wysokosc_d, sizeof(int));
	cudaMalloc((void**)&szerokosc_d, sizeof(int));

	// 2.Przepisanie dannych z CPU do GPU

	cudaMemcpy(piksele_Re_d, piksele_Re, sizeof(double)*rozmiar, cudaMemcpyHostToDevice);
	cudaMemcpy(piksele_Im_d, piksele_Im, sizeof(double)*rozmiar, cudaMemcpyHostToDevice);

	cudaMemcpy(wysokosc_d, &wysokosc, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(szerokosc_d, &szerokosc, sizeof(int), cudaMemcpyHostToDevice);


	// 3. Dzialanie GPU

	dim3 grid((szerokosc + 31) / 32, (wysokosc + 31) / 32, 1);
	dim3 threads(32, 32);


	DFT_GPU_wiersze << < grid, threads >> >(piksele_Re_d, piksele_Im_d, Im_2_d, Re_2_d, wysokosc_d, szerokosc_d);

	DFT_GPU_kolumny << < grid, threads >> >(Re_2_d, Im_2_d, Im_3_d, Re_3_d, wysokosc_d, szerokosc_d);


	cudaMemcpy(piksele_Re, Re_3_d, sizeof(double)*szerokosc*wysokosc, cudaMemcpyDeviceToHost);
	cudaMemcpy(piksele_Im, Im_3_d, sizeof(double)*szerokosc*wysokosc, cudaMemcpyDeviceToHost);


	maska(piksele_Re, piksele_Im, wysokosc, szerokosc);

	cudaMemcpy(piksele_Re_d, piksele_Re, sizeof(double)*wysokosc*szerokosc, cudaMemcpyHostToDevice);
	cudaMemcpy(piksele_Im_d, piksele_Im, sizeof(double)*wysokosc*szerokosc, cudaMemcpyHostToDevice);

	IDFT_GPU_wiersze << < grid, threads >> >(piksele_Re_d, piksele_Im_d, Im_4_d, Re_4_d, wysokosc_d, szerokosc_d);

	IDFT_GPU_kolumny << < grid, threads >> >(Re_4_d, Im_4_d, Im_5_d, Re_5_d, wysokosc_d, szerokosc_d);

	cudaMemcpy(piksele_Re, Re_5_d, sizeof(double)*szerokosc*wysokosc, cudaMemcpyDeviceToHost);
	cudaMemcpy(piksele_Im, Im_5_d, sizeof(double)*szerokosc*wysokosc, cudaMemcpyDeviceToHost);


	int g = 0;
	for (int j = 0; j < wysokosc; j++)
	{
		for (int i = 0; i < szerokosc; i++)
		{
			Input(i, j)->Red = sqrt(pow(piksele_Re[g], 2) + pow(piksele_Im[g], 2));
			Input(i, j)->Green = sqrt(pow(piksele_Re[g], 2) + pow(piksele_Im[g], 2));
			Input(i, j)->Blue = sqrt(pow(piksele_Re[g], 2) + pow(piksele_Im[g], 2));
			g++;
		}
	}


	zapisanie_obrazu(Input);


	// czyszczenie

	delete[] piksele_Re;
	delete[] piksele_Im;


	cudaFree(piksele_Re_d);
	cudaFree(piksele_Im_d);
	cudaFree(Re_2_d);
	cudaFree(Im_2_d);
	cudaFree(Re_3_d);
	cudaFree(Im_3_d);
	cudaFree(Re_4_d);
	cudaFree(Im_4_d);
	cudaFree(Re_5_d);
	cudaFree(Im_5_d);
	cudaFree(szerokosc_d);
	cudaFree(wysokosc_d);


	return 0;
}