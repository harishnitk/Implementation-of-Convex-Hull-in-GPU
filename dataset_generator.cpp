/*
classname dataset_generator
*/

#include<bits/stdc++.h>
#include<vector>
#include "wb.h"

using namespace std;

static char *base_dir;

struct Point
{
   int x;
   int y;
};

static void write_input_data(char *file_name, vector<Point> p, int length) {

   FILE *handle = fopen(file_name, "w");
   fprintf(handle, "%d", length);
   
   vector<Point> :: iterator it;
   for(it=p.begin();it!=p.end();it++) {
     fprintf(handle, "\n%d %d", (*it).x,(*it).y);
   }
   fflush(handle);
   fclose(handle);
}

/*
@method [create_dataset]
@description pass two parameter director or number of element
*/
void create_dataset(int datasetNum,int input_length)
{
   const char *dir_name = wbDirectory_create(wbPath_join(base_dir, datasetNum));
   char *input_file_name  = wbPath_join(dir_name, "input.raw");
   
   vector< Point > input(input_length);

   srand(time(NULL));
   int schedule_mod = input_length>100?3:input_length/4;
   
   for(int i=0;i<input_length;i++)
   {
   	  input[i].x = rand()%(input_length+10)-schedule_mod;
      input[i].y = rand()%(input_length+10)-schedule_mod;
   }
   
   write_input_data(input_file_name, input, input_length);
   
   input.clear();
}

int main()
{
	/*create Base directory*/
    base_dir = wbPath_join(wbDirectory_current(),"ConvexHull","Dataset");
    
    create_dataset(0, 6);
    create_dataset(1, 1024);
    create_dataset(2, 513);
    create_dataset(3, 10000);
    create_dataset(4, 50000);
    create_dataset(5, 100000);
    
    return 0;
}

