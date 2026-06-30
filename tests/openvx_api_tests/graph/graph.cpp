/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstring>
#include <iostream>
#include <chrono>

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <vx_ext_amd.h>

using namespace std;

#define ERROR_CHECK_STATUS(status)                                                              \
    {                                                                                           \
        vx_status status_ = (status);                                                           \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

#define ERROR_CHECK_OBJECT(obj)                                                                 \
    {                                                                                           \
        vx_status status_ = vxGetStatus((vx_reference)(obj));                                   \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0)
    {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

static int getEnvironmentVariable(const char * name, char * value, size_t valueSize)
{
#if _WIN32
    char text[1024] = { 0 };
    DWORD len = GetEnvironmentVariableA(name, text, (DWORD)sizeof(text));
    if ( len > 1) {
        value[len-1] = '\0';
        if(isdigit(value[0]) != 0)
            return atoi(value);
        else return 1;
    }
#else
    const char * text = getenv(name);
    if (text) {
        strncpy(value, text, valueSize);
        value[strlen(text)+1] = '\0';
        if(isdigit(value[0]) != 0)
            return atoi(value);
        else return 1;
    }
#endif
    return -1;
}

int main(int argc, char **argv)
{

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph)

    AgoGraphImportInfo info = { 0 };

    // Resolve the MIVisionX install prefix: INSTALL_PATH takes priority
    // (used by CI when building against a non-system prefix), then ROCM_PATH.
    char installLocation[1024] = { 0 };
    int isInstallSet = getEnvironmentVariable("INSTALL_PATH", installLocation, sizeof(installLocation));
    if (isInstallSet <= 0) {
        int isRocmSet = getEnvironmentVariable("ROCM_PATH", installLocation, sizeof(installLocation));
        if (isRocmSet <= 0) {
            printf("\nROCM PATH: NOT SET -- skipping GDF import test\n");
            vxReleaseGraph(&graph);
            vxReleaseContext(&context);
            return 0;
        }
        printf("\nROCM PATH:%s\n", installLocation);
    } else {
        printf("\nINSTALL PATH:%s\n", installLocation);
    }
    char gdfFile[2048];
    const char gdfLocation[] = "/share/mivisionx/samples/gdf/read-gdf-sample.gdf";
    snprintf(gdfFile, sizeof(gdfFile), "%s%s", installLocation, gdfLocation);
    printf("GDF load file:%s\n", gdfFile);

    FILE *file = fopen(gdfFile, "rb");
    if (file == NULL) {
        perror("Error opening file:");
        return -1;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    info.text = (char *)malloc(file_size + 1);
    if (info.text == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return 1;
    }

    size_t bytes_read = fread(info.text, 1, file_size, file);
    if (bytes_read != file_size) {
        perror("Error reading file");
        free(info.text);
        fclose(file);
        return 1;
    }
    info.text[file_size] = '\0';
    fclose(file);
    printf("File contents:\n%s\n", info.text);

    ERROR_CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_IMPORT_FROM_TEXT, &info, sizeof(info)));

    auto start = std::chrono::steady_clock::now();
    ERROR_CHECK_STATUS(vxProcessGraph(graph));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "STATUS: vxProcessGraph() took " << (elapsed_seconds.count()*1000.0f) << "msec (1st iteration)\n";

    start = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++){
        ERROR_CHECK_STATUS(vxProcessGraph(graph));
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = (end-start)/100;
    std::cout << "STATUS: vxProcessGraph() took " << (elapsed_seconds.count()*1000.0f) << "msec (AVG)\n";

    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    free(info.text);

    return 0;
}
