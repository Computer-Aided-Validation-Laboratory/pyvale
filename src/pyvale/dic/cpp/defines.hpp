#define TITLE(a) std::cout << std::string(122, '-') << std::endl; std::cout << std::string(60 - std::strlen(a)/2, '-') << " " << a << " " << std::string(60 - std::strlen(a)/2, '-') << std::endl; std::cout << std::string(122, '-') << std::endl;
#define INFO_OUT(a,b) std::cout.width(100); std::cout << std::left << "  - " << a; std::cout << b << std::endl;
#define DEBUGGER std::cout << __FILE__ << " " << __LINE__ << std::endl;
#ifdef CUDA
        #define CUDA_CALL(x) do { if((x) != cudaSuccess) {\
        printf("Error at %s:%d\n",__FILE__,__LINE__);\
        exit(EXIT_FAILURE);}} while(0)

        #define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
        printf("Error at %s:%d\n",__FILE__,__LINE__);\
        exit(EXIT_FAILURE);}} while(0)
#endif
