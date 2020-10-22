#include <iostream>

//#define __HIP_PLATFORM_HCC__

// hip header file
#include <xmmintrin.h>
#define __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_khr_nn.h>
#include <VX/vxu.h>
#include <vx_ext_amd.h>
#include <string>
#include <chrono>
#define DUMP_IMAGE  0
#define PRINT_OUTPUT 

#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return -1; } }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
	size_t len = strlen(string);
	if (len > 0) {
		printf("%s", string);
		if (string[len - 1] != '\n')
			printf("\n");
		fflush(stdout);
	}
}


inline int64_t clockCounter()
{
	return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
	return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

vx_status makeInputImage1(vx_context context, vx_image img, int width, int height, int mem_type, int pix_val)
{
	ERROR_CHECK_OBJECT((vx_reference)img);

	if (mem_type == VX_MEMORY_TYPE_HOST) {
		vx_uint8 image_data[width*height];

		for (int i= 0; i< height; i++ ){
			for (int j=0; j < width; j++) {
				image_data[i*width + j] = pix_val;
			}
		}
#ifdef PRINT_OUTPUT
		printf("Image1::\n");
		for (int i = 0; i< height ; i++, printf("\n"))
       {
               for(int j=0 ; j<width ; j++)
               {
                       printf("%d \t",image_data[i*width + j]);
               }
       }
#endif
		vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
		vx_imagepatch_addressing_t addr;
		addr.dim_x = width;
		addr.dim_y = height;
		addr.stride_x = 1;
		addr.stride_y = width;
		ERROR_CHECK_STATUS(vxCopyImagePatch(img, &rect, 0, &addr, image_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	} else {
		vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
		vx_map_id map_id;
		vx_imagepatch_addressing_t addrId;

		/*********************** TESTING IMAGE DATA TYPES FOR GPU ***********************/

		// vx_int16 * ptr;
		vx_uint8 * ptr;

		ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		for (int i= 0; i< height; i++ ){
			for (int j=0; j < width; j++) {
				ptr[i*width + j] = pix_val;
			}
		}
#ifdef PRINT_OUTPUT
		printf("Image1::\n");
		for (int i = 0; i< height ; i++, printf("\n"))
       {
               for(int j=0 ; j<width ; j++)
               {
                       printf("%d \t",ptr[i*width + j]);
               }
       }
#endif
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#if DUMP_IMAGE
		FILE *fp = fopen("input1.bin", "wb");
		ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		if (fp) fwrite(ptr, 1, width*height, fp);
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#endif
	}
	vxReleaseImage(&img);
	return VX_SUCCESS;
}

vx_status makeInputImage2(vx_context context, vx_image img, vx_uint32 width, vx_uint32 height, int mem_type, int outer_pix_val,int inner_pix_val)
{
	ERROR_CHECK_OBJECT((vx_reference)img);

	if (mem_type == VX_MEMORY_TYPE_HOST) {
		/*********************** TESTING IMAGE DATA TYPES FOR HOST ***********************/

		// vx_int16 image_data[width*height];
		vx_uint8 image_data[width*height];

		for (int i= 0; i< height; i++ ){
			for (int j=0; j < width; j++) {
				if ( i>=0.4*width && i < 0.6*width && j >= 0.2*height && j<0.8*height )
					image_data[i*width + j] = inner_pix_val;
				else
					image_data[i*width + j] = outer_pix_val;
			}
		}
#ifdef PRINT_OUTPUT
		printf("Image2::\n");
       for (int i = 0; i< height ; i++, printf("\n"))
       {
               for(int j=0 ; j<width ; j++)
               {
                       printf("%d \t",image_data[i*width + j]);
               }
       }
#endif
		vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
		vx_imagepatch_addressing_t addr;
		addr.dim_x = width;
		addr.dim_y = height;
		addr.stride_x = 1;
		addr.stride_y = width;
		ERROR_CHECK_STATUS(vxCopyImagePatch(img, &rect, 0, &addr, image_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	} else {
		vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
		vx_map_id map_id;
		vx_imagepatch_addressing_t addrId;

		/*********************** TESTING IMAGE DATA TYPES FOR GPU ***********************/

		// vx_int16 *ptr;
		vx_uint8 * ptr;

		ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		for (int i= 0; i< height; i++ ){
			for (int j=0; j < width; j++) {
				if ( i>=0.4*width && i < 0.6*width && j >= 0.2*height && j<0.8*height )
					ptr[i*width + j] = inner_pix_val;
				else
					ptr[i*width + j] = outer_pix_val;
			}
		}
#ifdef PRINT_OUTPUT
		printf("Image2::\n");
       for (int i = 0; i< height ; i++, printf("\n"))
       {
               for(int j=0 ; j<width ; j++)
               {
                       printf("%d \t",ptr[i*width + j]);
               }
       }
#endif
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#if DUMP_IMAGE
		FILE *fp = fopen("input2.bin", "wb");
		ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		if (fp) fwrite(ptr, 1, width*height, fp);
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#endif
	}
	vxReleaseImage(&img);
	return VX_SUCCESS;
}


int main(int argc, const char ** argv) {

	 // check command-line usage
    const size_t MIN_ARG_COUNT = 4;
    if(argc < MIN_ARG_COUNT){
   	 printf( "Usage: ./hipvx_sample <width> <height> <gpu=1/cpu=0> <image1 pixel value> <image2 outer pixel value> <image2 inner pixel value>\n" );
	 printf("\nOptional Arguments: <image1 pixel value> <image2 outer pixel value> <image2 inner pixel value>\n\n");
        return -1;
    }
	
	void *ptr[3] = {nullptr, nullptr, nullptr};     // hip
	int64_t freq = clockFrequency(), t0, t1;
	vx_image img1, img2, img_out;
	vx_uint32 width = atoi(argv[1]);
	vx_uint32 height = atoi(argv[2]);
	
	int affinity = atoi(argv[3]);           // 0 for CPU and 1 for GPU
	if (width <= 0) width = 100;
	if (height <= 0) height = 100;
	if (affinity <= 0) affinity = 0;
	char* outImgBuffer = new char[width*height];

	//image pixel values  user given
	int pix_img1, pix_outer_img2, pix_inner_img2;
	pix_img1 = (argc < 5) ?  100 : atoi(argv[4]);
	pix_outer_img2 =  (argc < 6) ?  100 : atoi(argv[5]);
	pix_inner_img2 =  (argc < 7) ?  100 : atoi(argv[6]);



	// create context, input, output, and graph
	vxRegisterLogCallback(NULL, log_callback, vx_false_e);
	vx_context context = vxCreateContext();
	vx_status status = vxGetStatus((vx_reference)context);
	if(status) {
		printf("ERROR: vxCreateContext() failed\n");
		return -1;
	}
	vxRegisterLogCallback(context, log_callback, vx_false_e);
	vx_graph graph = vxCreateGraph(context);
	// create graph and set affinity
	if (!affinity)
	{
		if (graph)
		{ 

			/*********************** TESTING IMAGE TYPES FOR HOST ***********************/

			img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
			// img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
			img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
			// img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
			img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
			// img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
			
			
			ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
			AgoTargetAffinityInfo affinity;
			affinity.device_type = AGO_TARGET_AFFINITY_CPU;
			affinity.device_info = 0;
			ERROR_CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));
		
			/*********************** TESTING KERNELS FOR HOST ***********************/

			vx_node node = vxAbsDiffNode(graph, img1, img2, img_out);
			// vx_node node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
			// vx_node node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
			// vx_node node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
			// vx_node node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
			// vx_node node = vxMultiplyNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
			// vx_node node =vxAndNode(graph, img1, img2, img_out);
			// vx_node node = vxOrNode(graph, img1, img2, img_out);
			// vx_node node = vxXorNode(graph, img1, img2, img_out);
			// vx_node node = vxNotNode(graph,img2, img_out); //Only Image 2 is used

			

		
			if (node)
			{
				status = vxVerifyGraph(graph);
				ERROR_CHECK_STATUS(makeInputImage1(context, img1, width, height, VX_MEMORY_TYPE_HOST, pix_img1));
				ERROR_CHECK_STATUS(makeInputImage2(context, img2, width, height, VX_MEMORY_TYPE_HOST, pix_outer_img2, pix_inner_img2));
				printf("After makeInputImage\n");
				if (status == VX_SUCCESS)
				{
					status = vxProcessGraph(graph);
				}
				printf("After ProcessGraph\n");
				vxReleaseNode(&node);
			}
			vxReleaseGraph(&graph);
		}
	}
	else
	{
		vx_imagepatch_addressing_t addr = { 0};
		addr.dim_x = width;
		addr.dim_y = height;
		addr.stride_x = 1;
		addr.stride_y = width;
		hipMalloc((void**)&ptr[0], width*height);
		hipMalloc((void**)&ptr[1], width*height);
		hipMalloc((void**)&ptr[2], width*height);
		hipMemset(ptr[2], 0, width*height);
    	// printf("Main: dst: %p src1: %p src2: %p <%dx%d>\n", ptr[2], ptr[0], ptr[1], width, height);

		/*********************** TESTING IMAGE TYPES FOR GPU-HIP ***********************/
		
		ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &addr, &ptr[0], VX_MEMORY_TYPE_HIP));
		// ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &addr, &ptr[0], VX_MEMORY_TYPE_HIP));
		ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &addr, &ptr[1], VX_MEMORY_TYPE_HIP));
		// ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &addr, &ptr[1], VX_MEMORY_TYPE_HIP));
		ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &addr, &ptr[2], VX_MEMORY_TYPE_HIP));
		// ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &addr, &ptr[2], VX_MEMORY_TYPE_HIP));
		
		if (graph)
		{
			AgoTargetAffinityInfo affinity;
			affinity.device_type = AGO_TARGET_AFFINITY_GPU;
			affinity.device_info = 0;

			ERROR_CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));
			



		/*********************** TESTING KERNELS FOR GPU ***********************/

			vx_node node = vxAbsDiffNode(graph, img1, img2, img_out);
			// vx_node node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
			// vx_node node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
			// vx_node node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
			// vx_node node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
			// vx_node node =vxAndNode(graph, img1, img2, img_out);
			// vx_node node = vxOrNode(graph, img1, img2, img_out);
			// vx_node node = vxXorNode(graph, img1, img2, img_out);
			// vx_node node = vxNotNode(graph,img2, img_out); //Only Image 2 is used
			


			
			if (node)
			{
				status = vxVerifyGraph(graph);
				ERROR_CHECK_STATUS(makeInputImage1(context, img1, width, height, VX_MEMORY_TYPE_HIP, pix_img1));
				ERROR_CHECK_STATUS(makeInputImage2(context, img2, width, height, VX_MEMORY_TYPE_HIP, pix_outer_img2, pix_inner_img2));
				if (status == VX_SUCCESS)
				{
					status = vxProcessGraph(graph);
				}
				vxReleaseNode(&node);
			}
			vxReleaseGraph(&graph);
		}
	}

	// check output values
	vx_rectangle_t rect = { 0, 0, width, height };
	vx_map_id  map_id;
	vx_imagepatch_addressing_t addr = {0};
	addr.stride_x = 1;
	addr.stride_y = width;
	
	
	
	/*********************** TESTING FOR DIFFERENT OUTPUT BIT DEPTHS ***********************/
	
	vx_uint8 *out_buf;
	// vx_int16 *out_buf;

	ERROR_CHECK_STATUS(vxMapImagePatch(img_out, &rect, 0, &map_id, &addr, (void **)&out_buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
	//ERROR_CHECK_STATUS(vxCopyImagePatch(img_out, &rect, 0, &addr, outImgBuffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	


	/*********************** VERIFYING KERNELS WITH EXPECTED OUTPUTS***********************/

	int h_rect = (int)(0.8*height)-(int)(0.2*height);
	int w_rect = (int)(0.6*width) -(int)(0.4*width);
	// int expected = 20*60*255;     // white only in roi 
	int expected =  ((h_rect*w_rect) * ((pix_img1 >= pix_inner_img2) ? (pix_img1 - pix_inner_img2) : (pix_inner_img2 - pix_img1)) ) + 
					(((width * height) - (h_rect*w_rect)) * ((pix_img1 >= pix_outer_img2) ? (pix_img1 - pix_outer_img2) : (pix_outer_img2 - pix_img1))); //AbsDiff_U8_U8U8
	// int expected =  ((h_rect*w_rect) * (pix_img1 + pix_inner_img2)) + (((width * height) - (h_rect*w_rect)) * (pix_img1 + pix_outer_img2)); //Add_U8_U8U8_Wrap & Add_U8_U8U8_Sat
	// int expected =  ((h_rect*w_rect) * (pix_img1 - pix_inner_img2)) + (((width * height) - (h_rect*w_rect)) * (pix_img1 - pix_outer_img2)); //Sub_U8_U8U8_Wrap & Sub_U8_U8U8_Sat

	// int expected =  ((h_rect*w_rect) * (pix_img1 & pix_inner_img2)) + (((width * height) - (h_rect*w_rect)) * (pix_img1 & pix_outer_img2)); // And_U8_U8U8
	// int expected =  ((h_rect*w_rect) * (pix_img1 | pix_inner_img2)) + (((width * height) - (h_rect*w_rect)) * (pix_img1 | pix_outer_img2)); // Or_U8_U8U8
	// int expected =  ((h_rect*w_rect) * (pix_img1 ^ pix_inner_img2)) + (((width * height) - (h_rect*w_rect)) * (pix_img1 ^ pix_outer_img2)); // Xor_U8_U8U8
	// int expected = ((h_rect*w_rect) * (255-pix_inner_img2)) + (((width * height) * (255-pix_outer_img2) )- ((h_rect*w_rect) * (255 - pix_outer_img2)));//Not_U8_U8U8
	int expected =  ((h_rect*w_rect) * ((pix_img1 >= pix_inner_img2) ? (pix_img1 - pix_inner_img2) : (pix_inner_img2 - pix_img1)) ) + 
(((width * height) - (h_rect*w_rect)) * ((pix_img1 >= pix_outer_img2) ? (pix_img1 - pix_outer_img2) : (pix_outer_img2 - pix_img1))); //AbsDiff_U8_U8U8
	
	/*********************** PRINT OUTPUT IMAGE ***********************/
#ifdef PRINT_OUTPUT
	 int i,j;
       for ( i = 0; i< height ; i++, printf("\n"))
       {
               for(j=0 ; j<width ; j++)
               {
                       printf("%d \t",out_buf[i*width + j]);
               }
       }
#endif

	
	int sum = 0;
	for (int i = 0; i < width*height; i++) {
		sum +=  out_buf[i];
	}
	if (sum != expected) {
		printf("FAILED: sum = %d expected = %d\n", sum, expected);
	} else {
		printf("PASSED: sum = %d expected = %d\n", sum, expected);
	}
#if DUMP_IMAGE
	FILE *fp = fopen("output.bin", "wb");
	if (fp) fwrite(out_buf, 1, width*height, fp);
#endif
	ERROR_CHECK_STATUS( vxUnmapImagePatch( img_out, map_id ) );

	// free the resources on host side
	if (ptr[0]) hipFree(ptr[0]);
	if (ptr[1]) hipFree(ptr[1]);
	if (ptr[2]) hipFree(ptr[2]);
	free(outImgBuffer);
	vxReleaseContext(&context);

	return 0;
}
