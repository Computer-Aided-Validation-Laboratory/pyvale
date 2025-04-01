A good site: https://enccs.github.io
Jargon
- Host - CPU
- Device - GPU
- FPGA - field-programmable gate array

# Overview of architecture approaches:
## Directive-based 
Annotate existing serial code with hints for how the compiler should parralelise. 
Uses a fork-join execution model:
![[threads.png]]
Serial until a parallel region is requests. Multiple threads are spawned, and the mast thread is in charge of the threads.
Implicit barrier at the end of parallel regions, where threads wait until all threads are complete. (If there is imbalanced workload, many threads will sit idle doing nothing.)
The same annotations can be used for compiling to CPU multithread or GPU easily (though performance will vary)

APIs available:
- OpenACC
- OpenMP

Pros:
- Incremental programming. Write serial, add parallelisation as necessary.
- Porting existing sequential code is easier
- Same code for device and host, just use compiler flags
- Low learning curve
- Good portability
Cons:
- Need to be mindful on underlying memory movement and what your commands are actually doing for performance.
- Lacks architecture-specific control (like local memory or const cache)

``` c++
#include <stdio.h>
#include <openacc.h>

#define NX 100000

int main(void)
{
    double vecA[NX], vecB[NX], vecC[NX];
    int i;

    /* Initialization of the vectors */
    for (i = 0; i < NX; i++) {
        vecA[i] = 1.0;
        vecB[i] = 2.0;
    }

    #pragma acc parallel loop gang(32), vector(16)    // OpenACC
	#pragma omp target teams distribute parallel for simd    // OpenMP
    for (i = 0; i < NX; i++) {
        vecC[i] = vecA[i] + vecB[i];
    }

    return 0;
}
```

Debugging. Use of gdb and other mature tools in multithreaded context. Some support for GPU-specific debuggers (like nv-gdb) for debugging OpenACC code on device.

Between OpenACC and OpenMP:
- OpenACC is more focused on accelerators (GPUs, FPGAs), whereas OMP includes multithreaded CPU.
- OpenMP is more prescriptive-  you need to specify what sort of parralelism you want. OpenACC is descriptive - the compiler figures it out.
- OpenMP has more features (good and bad)
Both used extensively in scientific computing.

Interop with Zig: None. Zig can compile this, but there will not be any acceleration. Some attempted projects at Zig-OpenMP compatibility

My conclusion: If you're writing c/c++ code and you want some performance improvements, stick on a pragma. Probably with OpenMP over OpenACC
## Non-portable Kernel-based
Write solely GPU-specific kernels. Main program executed on the CPU, which controls allocation, data transfer, and device launches. Because it is first and foremost for GPUs, there is access to  high-level tuning and advanced features (like shared memory).

Options:
- CUDA
- HIP

HIP handles the backend job of calling the associated CUDA API when running on NVidia cards, and ROCm when running on AMD.

HIP Pros:
- Open standards and portable between AMD and NVidia
- Similar API calls and structure.

HIP Cons:
 - Developing community (not as large and mature as CUDA)
 - Only GPU acceleration. Also limited hardware support for older devices.
 - High learning curve
 - Lacking in high-perf math libraries for common operations (compared to CUDA)
 - Differences in feature support with CUDA, particularly with newer or more specialized CUDA features.

Debugging: GPU-side debugger. Multithreaded is a pain to profile, even more so on a stochastic device like a GPU. Also has a perf profiller tool.

```C++
#include <cuda.h> // or #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void vector_add(float *A, float *B, float *C, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    C[tid] = A[tid] + B[tid];
  }
}

int main(void) {
  const int N = 100000;
  float *Ah, *Bh, *Ch, *Cref;
  float *Ad, *Bd, *Cd;
  int i;

  // Allocate the arrays on CPU
  Ah = (float *)malloc(N * sizeof(float));
  Bh = (float *)malloc(N * sizeof(float));
  Ch = (float *)malloc(N * sizeof(float));
  Cref = (float *)malloc(N * sizeof(float));

  // initialise data and calculate reference values on CPU
  for (i = 0; i < N; i++) {
    Ah[i] = sin(i) * 2.3;
    Bh[i] = cos(i) * 1.1;
    Cref[i] = Ah[i] + Bh[i];
  }

  // Allocate the arrays on GPU
  cudaMalloc((void **)&Ad, N * sizeof(float));
  cudaMalloc((void **)&Bd, N * sizeof(float));
  cudaMalloc((void **)&Cd, N * sizeof(float));
  // or for HIP
  hipMalloc((void **)&Ad, N * sizeof(float));

  // Transfer the data from CPU to GPU
  cudaMemcpy(Ad, Ah, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(Bd, Bh, sizeof(float) * N, cudaMemcpyHostToDevice);
  // Or HIP
  hipMemcpy(Bd, Bh, sizeof(float) * N, hipMemcpyHostToDevice);

  // define grid dimensions + launch the device kernel
  dim3 blocks, threads;
  threads = dim3(256, 1, 1);
  blocks = dim3((N + 256 - 1) / 256, 1, 1);

  // Launch Kernel
  vector_add<<<blocks, threads>>>(Ad, Bd, Cd, N);

  // copy results back to CPU
  cudaMemcpy(Ch, Cd, sizeof(float) * N, cudaMemcpyDeviceToHost);

  printf("reference: %f %f %f %f ... %f %f\n", Cref[0], Cref[1], Cref[2],
         Cref[3], Cref[N - 2], Cref[N - 1]);
  printf("   result: %f %f %f %f ... %f %f\n", Ch[0], Ch[1], Ch[2], Ch[3],
         Ch[N - 2], Ch[N - 1]);

  // confirm that results are correct
  float error = 0.0;
  float tolerance = 1e-6;
  float diff;
  for (i = 0; i < N; i++) {
    diff = fabs(Cref[i] - Ch[i]);
    if (diff > tolerance) {
      error += diff;
    }
  }
  printf("total error: %f\n", error);
  printf("  reference: %f at (42)\n", Cref[42]);
  printf("     result: %f at (42)\n", Ch[42]);

  // Free the GPU arrays
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);

  // Free the CPU arrays
  free(Ah);
  free(Bh);
  free(Ch);
  free(Cref);

  return 0;
}
```
replace `cuda` with `hip` above for exactly the same functionality.

A challenge with non-portable kernel-based code is the use of distinct memory regions for device and host, and managing the information between them.  Unified memory simplifies development, but under the hood, it still has data transfer like above when called.
```C++
  hipMallocManaged(&Ah, N * sizeof(float));
  hipMallocManaged(&Bh, N * sizeof(float));
```

Zig Interop: A multi-staged approach, but appears very possible.

My conclusion: A solid language choice. Similar language means a lots of transferable skills between CUDA and HIP/ROCm. Can be a pain to debug, but there are a suite of tools to assist.

## Portable Kernel-based
Higher abstraction. Aimed at reducing time and effort for deploying to multiple archs. Usually based on C++ and using lambda functions to define the loop/kernel body for running on multiple architectures and from different vendors. Code only needs to be written once for both CPU and GPU execution.

Open Standards:
- OpenCL
- Vulcan
- OpenGL
- SYCL
- (DirectX)
Ecosystems:
- Kokkos
- SYCL derivatives [Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html), [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp/), [triSYCL](https://github.com/triSYCL/triSYCL), and [ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home/).
- RAJA
- Alpaka

### OpenCL
```c
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000

static const char *programSource =
    "__kernel void vector_add(__global const float* A, __global const float* "
    "B, __global float* C, int N) {\n"
    "    int tid = get_global_id(0);\n"
    "    if (tid < N) {\n"
    "        C[tid] = A[tid] + B[tid];\n"
    "    }\n"
    "}\n";

int main() {
  // Initialize data and calculate reference values on CPU
  float Ah[N], Bh[N], Ch[N], Cref[N];
  for (int i = 0; i < N; i++) {
    Ah[i] = sin(i) * 2.3f;
    Bh[i] = cos(i) * 1.1f;
    Ch[i] = 12.f;
    Cref[i] = Ah[i] + Bh[i];
  }

  // Use the default device
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);
  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  // Build the kernel from string
  cl_program program =
      clCreateProgramWithSource(context, 1, &programSource, NULL, NULL);
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

  // Allocate the arrays on GPU
  cl_mem d_A =
      clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, NULL);
  cl_mem d_B =
      clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, NULL);
  cl_mem d_C =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, NULL);

  clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, N * sizeof(float), Ah, 0, NULL,
                       NULL);
  clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, N * sizeof(float), Bh, 0, NULL,
                       NULL);

  // Set arguments and launch the kernel
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
  cl_int N_as_cl_int = N;
  clSetKernelArg(kernel, 3, sizeof(cl_int), &N_as_cl_int);
  size_t globalSize = N;
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL,
                         NULL);

  // Copy the results back
  clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, N * sizeof(float), Ch, 0, NULL,
                      NULL);

  // Print reference and result values
  printf("Reference: %f %f %f %f ... %f %f\n", Cref[0], Cref[1], Cref[2],
         Cref[3], Cref[N - 2], Cref[N - 1]);
  printf("Result   : %f %f %f %f ... %f %f\n", Ch[0], Ch[1], Ch[2], Ch[3],
         Ch[N - 2], Ch[N - 1]);

  // Compare results and calculate the total error
  float error = 0.0f;
  float tolerance = 1e-6f;
  for (int i = 0; i < N; i++) {
    float diff = fabs(Cref[i] - Ch[i]);
    if (diff > tolerance) {
      error += diff;
    }
  }

  printf("Total error: %f\n", error);
  printf("Reference:   %f at (42)\n", Cref[42]);
  printf("Result   :   %f at (42)\n", Ch[42]);

  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseMemObject(d_C);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
```

OpenCL has a curious compilation process. Compiling: online and offline. 
- Online compiles when starting to run. It dynamically picks up the GPU architecture it will use and optimise for it. 
- Offline compiles binaries beforehand (when coding), but reduces portability as it might not be compiled for the accelerator used. 

Kernels are written in strings, so can compile with standard c compilers. Like with HIP, you write kernel code for the accelerator, and main-threaded code for the host.

Used in:
- Adobe photoshop
- GROMACS (molecular dynamics)
- [FluidX3D](https://github.com/ProjectPhysX/FluidX3D)
- Handbrake video processor
Pros:
- Cross compatible with different GPUs, CPUs, accellerators
- Can optimise over workers/threads/cores online, so more optimised performance across different devices
- Works on CPU, GPU, FPGA and more
Cons:
- Text string kernels?!!?
- Complex learning curve
- Still want to implement vendor-specific optimisations to make the fastest programs

### SYCL
An open standard for implementing accelerated computing. It is implemented in Intel's OneAPI and AdaptiveCpp. The idea is that you can write SYCL code, independent of the tool chain. However, different implementations have different compilers (and hence optimisations), so performance will vary from toolchain choices. Historically it is a more modern version of OpenCL. Kind of like how Vulkan is to OpenGL, both still have their place, but this is a new version that aims to target modern developments.

There are 2 main implementations to choose between, OneAPI and AdaptiveCpp.
The story between OneAPI and AdaptiveCpp feels similar to NVidia/HIP. Intel's version is more powerful, still cross compatible, but with the strongest performance on Intel, as well as the most modern features and implementations of the SYCL standard. AdaptiveCpp is the community-driven version that focuses on portability and compatibility.

```c++
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
  // Create an in-order queue
  sycl::queue q{sycl::property::queue::in_order()};

  // Print the device name, just for fun
  std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  const int n = 1024; // Vector size

  // Allocate device and host memory for the first input vector
  float *d_x = sycl::malloc_device<float>(n, q);
  float *h_x = sycl::malloc_host<float>(n, q);

  // Allocate second input vector on device and host, d_y and h_y
  float *d_y = sycl::malloc_device<float>(n, q);
  float *h_y = sycl::malloc_host<float>(n, q);

  // Allocate device and host memory for the output vector
  float *d_z = sycl::malloc_device<float>(n, q);
  float *h_z = sycl::malloc_host<float>(n, q);

  // Initialize values on host
  for (int i = 0; i < n; i++) {
    h_x[i] = i;
    h_y[i] = n - i;
  }

  const float alpha = 0.42f;

  q.copy<float>(h_x, d_x, n);
  q.copy<float>(h_y, d_y, n);

  // Don't need to wait before using the data because we are using an in-order queue. The in-order queue guarantees that commands are executed in the order they are submitted. So, the copy operations will finish before the kernel is executed.

  // Run the kernel
  q.parallel_for(sycl::range{n}, [=](sycl::id i) {
    d_z[i] = alpha * d_x[i] + d_y[i];
  });

  // Copy d_z to h_z
  q.copy<float>(d_z, h_z, n);

  // Wait for the copy to complete
  q.wait();

  // Check the results
  bool ok = true;
  for (int i = 0; i < n; i++) {
    float ref = alpha * h_x[i] + h_y[i]; // Reference value
    float tol = 1e-5;                    // Relative tolerance
    if (std::abs((h_z[i] - ref)) > tol * std::abs(ref)) {
      std::cout << i << " " << h_z[i] << " " << h_x[i] << " " << h_y[i]
                << std::endl;
      ok = false;
      break;
    }
  }

  if (ok)
    std::cout << "Results are correct!" << std::endl;
  else
    std::cout << "Results are NOT correct!" << std::endl;

  // Free allocated memory
  sycl::free(d_x, q);
  sycl::free(h_x, q);
  sycl::free(d_y, q);
  sycl::free(h_y, q);
  sycl::free(d_z, q);
  sycl::free(h_z, q);

  return 0;
}
```

Or an alternative, more kernel-centric version:

``` C++
#include <iostream>
#include <sycl/sycl.hpp>
 
void add_vectors(sycl::queue& queue, sycl::buffer<float>& a, sycl::buffer<float>& b, sycl::buffer<float>& c) {
   sycl::range n(a.size());
 
   queue.submit([&](sycl::handler& cgh) {
      auto in_a = a.get_access<sycl::access::mode::read>(cgh);
      auto in_b = b.get_access<sycl::access::mode::read>(cgh);
      auto out_c = c.get_access<sycl::access::mode::write>(cgh);
 
      cgh.parallel_for<class add_vectors>(n, [=](sycl::id<1> i) {
               out_c[i] = in_a[i] + in_b[i];
      });
   });
}
 
int main(int, char**) {
   const size_t n = 100;
 
   std::vector<float> a(n, 1.0f);
   std::vector<float> b(n, 2.0f);
   std::vector<float> c(n, 0.0f);
 
   sycl::buffer<float> a_buf{a};
   sycl::buffer<float> b_buf{b};
   sycl::buffer<float> c_buf{c};
 
   sycl::queue q;
 
   add_vectors(q, a_buf, b_buf, c_buf);
 
   auto result = c_buf.get_access<sycl::access::mode::read>();
   for (size_t i = 0; i < n; ++i) {
      std::cout << result[i] << " ";
   }
 
   return 0;
}
```

My thoughts: I like the vendor-agnostic implementation of SYCL. Tries to modernise OpenCL, including reducing boilerplate, while maintaining the option for fine-grain control. It also has a better separation of kernel and host code.

Used in:
- Self-driving cars
- A lot of backing from Intel
Pros:
- C++
- Community driven (adaptivecpp) or intensive industry backing (OneAPI)
- Portable to multitude of architectures, like OpenCL
Cons:
- C++
- Newer technology compared to CUDA and OpenCL (but similar to HIP)
- Smaller community
- [A little slower than OpenCL or OpenMP on the CPU](https://www.researchgate.net/publication/360501361_A_Comparison_of_SYCL_OpenCL_CUDA_and_OpenMP_for_Massively_Parallel_Support_Vector_Machine_Classification_on_Multi-Vendor_Hardware) 



## Honorary mentions

### Julia
Built in support for nvidia & AMD GPU through libraries for each vendor.
```Julia
using CUDA

function vector_add_kernel(d_result, d_a, d_b, n::Int32)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds d_result[idx] = d_a[idx] + d_b[idx]
    end
    return nothing
end

n = 1024
h_a = rand(Float32, n)
h_b = rand(Float32, n)
h_c = rand(Float32, n)
h_d = rand(Float32, n)
d_a = CuArray(h_a)
d_b = CuArray(h_b)
d_c = CuArray(h_c)
d_d = CuArray(h_d)
d_result1 = CuArray{Float32}(undef, n)
d_result2 = CuArray{Float32}(undef, n)

threads = 256
blocks = cld(n, threads)

@cuda threads=threads blocks=blocks vector_add_kernel(d_result1, d_a, d_b, Int32(n))
@cuda threads=threads blocks=blocks vector_add_kernel(d_result2, d_c, d_d, Int32(n))

h_result1 = Array(d_result1)
h_result2 = Array(d_result2)

result1_cpu = h_a + h_b
result2_cpu = h_c + h_d
println("Results 1 match: ", h_result1 ≈ result1_cpu)
println("Results 2 match: ", h_result2 ≈ result2_cpu)
```
Can swap `using CUDA` for AMD at the top to change architecture functionality.

I like the fact that GPU support and multi threading is built into the language. It's a quite neat alternative to python for high-performance scientific computing
### Vulkan

I'll just link to the code here, because there is a _lot_ of boilerplate (the script totals 548 lines!): https://github.com/kbenzie/vulkan-examples/blob/master/vector_add/vector_add.cpp

I would consider writing in Vulkan if self-flagellation felt too kind.

Final thoughts
If you want to code in C: OpenMP or OpenCL