This project is inspired by [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) which is designed for real-time raytracing.
LusiaCompute is an excellent framework which use c++ dsl to generate backend shader which can run on CUDA, Metal, DX12 and really easy to extent with plugins.

But LuisaForce is focus on GPU physics simulation which will have different design target. So I reorganize LuisaCompute code into this project.
The main different is that LuisaCompute will use graphics API like DX12, Metal and Vulkan in the future.
But this project will only use hardware-related API like CUDA, Metal, ROCm and SYCL, because only these API will provide high performance tensor and BLAS operation library.
So when use LuisaCompute, user should insert device name into context to create device, in one compute, there will be cuda, dx12, vulkan in the same time. 
But in this project, I assume there are only one kind of device, so there is no need to choose one.

And I remove all ray-tracing related API and will add physics related data structure like vdb, hash-grid and bvh later.
The different design goal make me use a separate project instead use extension in LuisaCompute. But I still keep most of good design of it.

## Feature
1. Use hardware-related API like CUDA, Metal, ROCm and SYCL
2. Data Structure used by physics simulation like hash-grid, vdb and bvh.
3Use USD as main exchangeable format and render by hydra-storm