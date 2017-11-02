#include "wb.h"
#include<bits/stdc++.h>
#include<vector>
#include<fstream>
#include<string.h>
#include<sstream>
#include<stdio.h>

using namespace std;

typedef pair<long int,long int> Point;
Point *hull;

#define CUDA_CHECK(ans)                                                   \
{ gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}


/*
@ region Definition of global kernel function
*/

__global__ void lowerHull(Point *input, long int *label, long int *distance ,Point *devHull,long int size)
{
   long int Idx = blockIdx.x*blockDim.x+threadIdx.x;
   
   /*
   @ calculate perpendicular point
   */
   if((Idx)<size)
   {

     Point P = input[Idx];
     long lb = label[Idx];
     Point min = devHull[lb];
     Point max = devHull[lb+1];
     distance[Idx] = (P.second-min.second)*(max.first-min.first)-(max.second-min.second)*(P.first-min.first);
   }

}

__global__ void segmentedScan(Point *dist, Point *label, long int size)
{
    long int Idx = blockIdx.x*blockDim.x+threadIdx;
}

/*
@endregion
*/


int main(int argc, char *argv[]) {
  
  long int inputLength,itr;
  Point leftmost_point{INT_MAX,0},rightmost_point{INT_MIN,0};
  
  wbTime_start(Generic, "Importing data and creating memory on host");
  ifstream file;
  file.open(argv[2]);
  file>>inputLength;
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  /*
  @region declaration
  */
  Point *hostInput = new Point[inputLength];
  Point *deviceInput = new Point[inputLength];;
  hull = new Point[inputLength];
  Point *deviceHull = new Point[inputLength];
  long int *Label = new long int[inputLength];
  long int *Distance = new long int[inputLength];
  long int *deviceLabel = new long int[inputLength];;
  long int *deviceDistance = new long int[inputLength];; 
  /*
  @endregion
  */

  memset(Label,0,inputLength*sizeof(long int));
  memset(Distance,0,inputLength*sizeof(long int));
 
  for(itr=0;itr<inputLength;itr++)
  {
     file>>hostInput[itr].first;
     file>>hostInput[itr].second;
  }
  file.close();

  int threads_per_block = 512;
  dim3 blocks(ceil(inputLength/threads_per_block)+1,1,1);
  
  /*
  @ param find the leftmost and rightmost Point in x direction
  */
  for(itr=0;itr<inputLength;itr++)
  {
     if(leftmost_point.first>hostInput[itr].first)
     {
        leftmost_point = hostInput[itr];
     }
     if(rightmost_point.first<hostInput[itr].first)
     {
        rightmost_point = hostInput[itr];
     }
  }

  /*
  @region Memory-Allocation
  */
  cudaMalloc((void **)&deviceInput,inputLength*sizeof(Point));
  cudaMemcpy(deviceInput,hostInput,inputLength*sizeof(Point),cudaMemcpyHostToDevice);
  
  //--------------- insert point in hull -----------//
  hull[0] = leftmost_point;
  hull[1] = rightmost_point;
  

  cudaMalloc((void **)&deviceHull,inputLength*sizeof(Point));
  cudaMemcpy(deviceHull,hull,inputLength*sizeof(Point),cudaMemcpyHostToDevice);
  cudaMalloc((void **)&deviceLabel,inputLength*sizeof(long int));
  cudaMemcpy(deviceLabel,Label,inputLength*sizeof(long int),cudaMemcpyHostToDevice);
  cudaMalloc((void **)&deviceDistance,inputLength*sizeof(long int));
  cudaMemcpy(deviceDistance,Distance,inputLength*sizeof(long int),cudaMemcpyHostToDevice); 
  
  /*
  @endregion
  */

  /*
  @ param calculate the LowerHull
  */
  
  lowerHull<<<blocks,threads_per_block>>>(deviceInput,deviceLabel,deviceDistance,deviceHull,inputLength);
  //upperHull<< blocks , threads_per_block >>( deviceInput, deviceLabel, deviceDistance, inputLength ); 
  //cudaMemcpy(Distance,deviceDistance,inputLength*sizeof(Point),cudaMemcpyDeviceToHost);
  
  segmentedScan<<<blocks , threads_per_block >>>(deviceDistance,deviceLabel,inputLength);
  
  /*
  @endregion
  */

  cudaDeviceSynchronize();
  

  return 0;
}
