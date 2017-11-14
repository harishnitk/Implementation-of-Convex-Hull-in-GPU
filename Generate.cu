#include "wb.h"
#include<bits/stdc++.h>
#include<vector>
#include<fstream>
#include<string.h>
#include<sstream>
#include<stdio.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>
#include<thrust/scan.h>
#include <thrust/sort.h>
#include<vector>
#include<climits>
#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include<cuda_runtime.h>
#include<thrust/extrema.h>

using namespace std;

typedef pair<long int,long int> Point;

set<Point> hull;


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
***@ global function region
*/

__global__ void Launch_convexHull(Point *dPoints,long int psize,long int *dcheck,long int index,int *inc,int *dinc)
{
   int idx = blockIdx.x*blockDim.x+threadIdx.x;
   
   if(idx<psize){
   
     Point p = dPoints[index];
     Point q = dPoints[*inc];
     Point r = dPoints[dcheck[*dinc]];
     long int val = (q.second - p.second)*(r.first-q.first)-(q.first-p.first)*(r.second-q.second);
  
     __syncthreads();
     
     if(val<0)
     {

        atomicAdd(dinc,1);
        dcheck[*dinc] = *inc;
     }
     
     atomicAdd(inc,1);
     
     __syncthreads();
   
   }
}


int main(int argc, char *argv[]) {
  
  long int inputLength,itr;
  
  wbTime_start(Generic, "Importing data and creating memory on host");
  ifstream file;
  file.open(argv[2]);
  file>>inputLength;
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  /*
  @ declaration section
  */
  Point *hostPoints = new Point[inputLength];
  Point *devicePoints = new Point[inputLength];
  long int *check_orientation_point = new long int[inputLength];
  long int *device_check_orientation_point = new long int[inputLength];


  for(itr=0;itr<inputLength;itr++)
  {
     file>>hostPoints[itr].first;
     file>>hostPoints[itr].second;
  }
  file.close();
  /*
  @ region end
  */

  long int start = 0;
  for(itr=1;itr<inputLength;itr++)
  {
     if(hostPoints[itr].first<hostPoints[start].first)
     {
           start = itr;
     }
  }
  
  long int point = start,next;

  //Memory Allocation in GPU
  cudaMalloc((void **)&devicePoints,sizeof(Point)*inputLength);
  cudaMemcpy(devicePoints,hostPoints,sizeof(Point)*inputLength,cudaMemcpyHostToDevice);
  
  int threads_per_block = 512;
  dim3 blocks(ceil(inputLength/threads_per_block)+1,1,1);
  
  int z=0;
  do
  {
      hull.insert(hostPoints[point]);
 
      next = (point+1)%inputLength;
      
      memset(check_orientation_point,-1,sizeof(long int)*inputLength);
      check_orientation_point[0] = next;

      cudaMalloc((void **)&device_check_orientation_point,sizeof(long int)*inputLength);
      cudaMemcpy(device_check_orientation_point,check_orientation_point,sizeof(long int)*inputLength,cudaMemcpyHostToDevice);
      
      int *inc = (int *)malloc(sizeof(int));
      int *c_inc = (int *)malloc(sizeof(int));

      int *d_inc,*d_c_inc;
      cudaMalloc((void **)&d_inc,sizeof(int));
      cudaMemcpy(d_inc,inc,sizeof(int),cudaMemcpyHostToDevice);
      cudaMalloc((void **)&d_c_inc,sizeof(int));
      cudaMemcpy(d_c_inc,c_inc,sizeof(int),cudaMemcpyHostToDevice);


      Launch_convexHull<<<blocks,threads_per_block>>>(devicePoints,inputLength,device_check_orientation_point,point,d_inc,d_c_inc);
      
      cudaMemcpy(check_orientation_point,device_check_orientation_point,sizeof(long int)*inputLength,cudaMemcpyDeviceToHost);
      cudaMemcpy(c_inc,d_c_inc,sizeof(int),cudaMemcpyDeviceToHost); 
      cudaMemcpy(inc,d_inc,sizeof(int),cudaMemcpyDeviceToHost); 

      //do reduction
      
      point = check_orientation_point[*c_inc-1];

      z++;
  }while(z!=10);//while(point!=start);
  
  set<Point>::iterator it;
  
  for (it = hull.begin();it!=hull.end(); ++it)
  {
     cout<<it->first<<" "<<it->second<<endl;
  }

  cudaDeviceSynchronize();
  

  return 0;
}