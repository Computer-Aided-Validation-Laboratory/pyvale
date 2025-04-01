// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// cpp header files
#include <cuda.h>
#include <cstring>
#include <curand.h>
#include <iostream>
#include <cuda_runtime.h>

// STD library Header files
#include "../cpp/defines.hpp"

namespace cuglobal {


    //gpu variables
	int tpb;
	int bpg;


    void device_info(int n_ss){
            
        TITLE("CUDA DEVICE PROPERTIES");
        
        
        // Get number of available GPU devices
        int devicesCount;
        cudaGetDeviceCount(&devicesCount);
        INFO_OUT("Number of available GPU devices:", devicesCount);	
    
        std::string devicename_str;
        struct cudaDeviceProp properties;
        
        for (int i = 0; i < devicesCount; i++){
            cudaGetDeviceProperties(&properties, i);
            devicename_str = "Name of device " + std::to_string(i) + ":";	
            INFO_OUT(devicename_str, properties.name);
        }
        
        // check if device has been selected in config file	
        int device = 0;	
        if ((devicesCount != 1)) {
            std::cout << "More than one GPU has been found. No Specific GPU has been selected. \n";
            std::cout << "GPU has been automatically assigned to device 0. \n";
        }

        cudaSetDevice(device);
        cudaGetDeviceProperties(&properties, device);
        
        // print GPU information to stdout
        INFO_OUT("Device name:", properties.name);
        INFO_OUT("Memory Clock Rate (KHz):", properties.memoryClockRate);
        INFO_OUT("Memory Bus Width (bits):", properties.memoryBusWidth);
        INFO_OUT("Peak Memory Bandwidth (GB/s):", 2.0*properties.memoryClockRate*(properties.memoryBusWidth/8)/1.0e6);
        INFO_OUT("Total Device Global Memory (Mb):", properties.totalGlobalMem/1024/1024)
        INFO_OUT("Total Device Constant Memory (Kb):", properties.totalConstMem/1024)
        INFO_OUT("Device major.minor:", properties.major << "." << properties.minor)
        INFO_OUT("Device registers per block:", properties.regsPerBlock);
        INFO_OUT("multiprocessors:", properties.multiProcessorCount);
        INFO_OUT("max threads per processor:", properties.maxThreadsPerMultiProcessor);
        INFO_OUT("max threads per block:", properties.maxThreadsPerBlock);	
        
        tpb =  256;
        bpg = (n_ss + tpb - 1) / tpb;

        INFO_OUT("Simulation Threads Per Block:", tpb);
        INFO_OUT("Simulation Blocks Per grid:", bpg);
        
    }

    // void clear_memory(){
    //     CUDA_CALL(cudaFree(dSx1d));
    //     std::cout << "Deallocated GPU memory."
    // }


    // void allocate_memory(){

    //     //Device spin variables
    //     CUDA_CALL(cudaMalloc((void**)&dSx1d, sizeof(double)*params::Nspins));
    //     CUDA_CALL(cudaMemset(cuheun::Sdashnx, 0.0, sizeof(double) * params::Nspins));

    // }

    // void copy_to_gpu(){
	// 	CUDA_CALL(cudaMemset(cuheun::Sdashnx, 0.0, sizeof(double) * params::Nspins));

    // }

    // void copy_to_cpu
	// 	CUDA_CALL(cudaMemset(cuheun::Sdashnx, 0.0, sizeof(double) * params::Nspins));
    // }

}