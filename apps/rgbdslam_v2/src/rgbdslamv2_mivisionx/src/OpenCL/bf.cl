__kernel void bruteforce_hamming(__global const unsigned long int *query_value, __global const unsigned long int *search_array, const int query_size, const int search_size, __global int *min_distance, __global int *result_index, __global int *return_value)
{
  // Get global ID of thread
  const int idx = get_global_id(0);
  // Extract position in query_value and search_array from global_id
  unsigned int i = idx / search_size;
  unsigned int j = idx % search_size;

  if((i < query_size) && (j < search_size))
  {
      // Return value is hamming distance
      // Feature size is 8*32 bits = 4*64 bits (unsigned long int in OpenCL - 64 bits)
      return_value[i * search_size + j] = popcount(query_value[(4 * i) + 0] ^ search_array[(4 * j) + 0]) 
                                        + popcount(query_value[(4 * i) + 1] ^ search_array[(4 * j) + 1]) 
                                        + popcount(query_value[(4 * i) + 2] ^ search_array[(4 * j) + 2]) 
                                        + popcount(query_value[(4 * i) + 3] ^ search_array[(4 * j) + 3]);
      // Assign maximum possible value to min_distance
      min_distance[i] = 256 + 1;
      result_index[i] = -1;
  }
 
}

__kernel void bruteforce_min(const int query_size, const int search_size, __global int *min_distance, __global int *result_index, __global int *return_value)
{
  const int idx = get_global_id(0);
  unsigned int i = idx;
  // Find minimum distance corresponding to each element in query_value and save the index as well
  if(i < query_size)
  {
    for(int j = 0; j < search_size; j++)
    {
      if(return_value[i * search_size + j] < min_distance[i])
      {
        min_distance[i] = return_value[i * search_size + j];
        result_index[i] = j;
      }
    }
  }
}

__kernel void bruteforce_match(const int query_size, const int search_size, __global int *min_distance, __global int *result_index, __global int *match_array)
{
  const int idx = get_global_id(0);
  unsigned int i = idx;
  // Write all values to match_array, which will be passed back in to the host program
  if(i < query_size)
  {
    if(min_distance[i] < 128) 
    {
      match_array[i * 3 + 0] = i;
      match_array[i * 3 + 1] = result_index[i];
      match_array[i * 3 + 2] = min_distance[i];
    }
  }
}
