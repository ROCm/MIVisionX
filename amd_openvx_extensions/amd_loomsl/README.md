# Radeon Loom Stitching Library (vx_loomsl)
Radeon Loom Stitching Library (beta preview) is a highly optimized library for 360 degree video stitching applications. This library consists of:
* *Live Stitch API*: stitching framework built on top of OpenVX kernels (see [live_stitch_api.h](live_stitch_api.h) for API)
* *OpenVX module* [***vx_loomsl***]: additional OpenVX kernels needed for 360 degree video stitching

The [loom_shell](../../utilities/loom_shell/README.md) command-line tool can be used to build your application quickly. It provides direct access to Live Stitch API by encapsulating the calls to enable rapid prototyping.

This software is provided under a MIT-style license,  see the file COPYRIGHT.txt for details.

## Features
* Real-time live 360 degree video stitching optimized for Radeon Pro Graphics
* Upto 31 cameras
* Upto 7680x3840 output resolution
* RGB and YUV 4:2:2 image formats
* Overlay other videos on top of stitched video
* Support for 3rd party *LoomIO* plug-ins for camera capture and stitched output
* Support PtGui project export/import for camera calibration

## Live Stitch API: Simple Example
Let's consider a 360 rig that has 3 1080p cameras with Circular FishEye lenses. 
The below example demonstrates how to stitch images from these cameras into a 4K Equirectangular buffer.

````
    #include "vx_loomsl/live_stitch_api.h"
    #include "utils/loom_shell/loom_shell_util.h"

    int main()
    {
      # define camera orientation and lens parameters
      camera_params cam1_par = { { 120,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177} };
      camera_params cam2_par = { {   0,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177} };
      camera_params cam3_par = { {-120,0,90,0,0,0},{176,1094,547,0,-37,ptgui_lens_fisheye_circ,-0.1719,0.1539,1.0177} };

      # create a live stitch instance and initialize
      ls_context context;
      context = lsCreateContext();
      lsSetOutputConfig(context,VX_DF_IMAGE_RGB,3840,1920);
      lsSetCameraConfig(context,3,1,VX_DF_IMAGE_RGB,1920,1080*3);
      lsSetCameraParams(context, 0, &cam1_par);
      lsSetCameraParams(context, 1, &cam2_par);
      lsSetCameraParams(context, 2, &cam3_par);
      lsInitialize(context);

      # Get OpenCL context and create OpenCL buffers for input and output
      cl_context opencl_context;
      cl_mem buf[2];
      lsGetOpenCLContext(context,&opencl_context);
      createBuffer(opencl_context,3*1920*1080*3, &buf[0]);
      createBuffer(opencl_context,3*3840*1920  , &buf[1]);

      # load CAM00.bmp, CAM01.bmp, and CAM02.bmp (1920x1080 each) into buf[0]
      loadBufferFromMultipleImages(buf[0],"CAM%02d.bmp",3,1,VX_DF_IMAGE_RGB,1920,1080*3);

      # set input and output buffers and stitch a frame
      lsSetCameraBuffer(context, &buf[0]);
      lsSetOutputBuffer(context, &buf[1]);
      lsScheduleFrame(context);
      lsWaitForCompletion(context);

      # save the stitched output into "output.bmp"
      saveBufferToImage(buf[1],"output.bmp",VX_DF_IMAGE_RGB,3840,1920);

      # release resources
      releaseBuffer(&buf[0]);
      releaseBuffer(&buf[1]);
      lsReleaseContext(&context);
      
      return 0;
    }
````
## Live Stitch API: Real-time Live Stitch using LoomIO
This example makes use of a 3rd party LoomIO plug-ins for live camera capture and display.

````
    #include "vx_loomsl/live_stitch_api.h"
    int main()
    {
        // create context, configure, and initialize
        ls_context context;
        context = lsCreateContext();
        lsSetOutputConfig(context, VX_DF_IMAGE_RGB, 3840, 1920);
        lsSetCameraConfig(context, 16, 1, VX_DF_IMAGE_RGB, 1920, 1080 * 16);
        lsImportConfiguration(context, "pts", "myrig.pts");
        lsSetCameraModule(context, "vx_loomio_bm", "com.amd.loomio_bm.capture", "30,0,0,16");
        lsSetOutputModule(context, "vx_loomio_bm", "com.amd.loomio_bm.display", "30,0,0");
        lsInitialize(context);

        // process live from camera until aborted by input capture plug-in
        for(;;) {
          vx_status status;
          status = lsScheduleFrame(context);
          if (status != VX_SUCCESS) break;
          status = lsWaitForCompletion(context);
          if (status != VX_SUCCESS) break;
        }

        // release the context
        lsReleaseContext(&context);

        return 0;
    }
````
