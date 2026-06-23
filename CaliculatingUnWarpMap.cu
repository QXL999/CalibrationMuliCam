// cuda_phase_unwrap.cu
#include <cuda_runtime.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>

#define CV_PI_F 3.14159265358979323846f
#define TWO_PI_F (2.0f * CV_PI_F)

// 将 CPU 的 Mat 转换为 float 指针并上传
// 该文件只包含内核函数和宿主端调用接口

// -------------------------------------------------------------------
// 1. 四步相移计算包裹相位 (输出范围 [-π, π])
// -------------------------------------------------------------------
__global__ void computeWrappedPhaseKernel(
    const float* I1, const float* I2, const float* I3, const float* I4,
    float* phase, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    int idx = y * cols + x;

    float i1 = I1[idx], i2 = I2[idx], i3 = I3[idx], i4 = I4[idx];
    float num = i4 - i2;
    float den = i1 - i3;
    float val = atan2f(num, den);
    phase[idx] = val;
}

// -------------------------------------------------------------------
// 2. 安全相位差 (模 2π，输出 [0, 2π))
// -------------------------------------------------------------------
__global__ void safePhaseDifferenceKernel(
    const float* phi1, const float* phi2, float* result,
    int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    int idx = y * cols + x;

    float p1 = phi1[idx], p2 = phi2[idx];
    result[idx] = (p1 >= p2) ? (p1 - p2) : (p1 + TWO_PI_F - p2);
}

// -------------------------------------------------------------------
// 3. 三频外差分层展开 (竖直或水平条纹)
//    输入: phi1, phi2, phi3 (三个频率的包裹相位, 范围 [-π, π])
//    输出: unwrapped1 (最高频率的绝对相位, 单位弧度)
// -------------------------------------------------------------------
__global__ void unwrapMultiFreqKernel(
    const float* phi1, const float* phi2, const float* phi3,
    float* unwrapped1,
    float lambda1, float lambda2, float lambda3,
    float lambda12, float lambda23, float lambda123,
    int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    int idx = y * cols + x;

    // 转换为 [0, 2π) 以进行外差减法
    float ph1 = phi1[idx];
    float ph2 = phi2[idx];
    float ph3 = phi3[idx];
    // 归一化到 [0, 2π)
    ph1 = (ph1 < 0.0f) ? (ph1 + TWO_PI_F) : ph1;
    ph2 = (ph2 < 0.0f) ? (ph2 + TWO_PI_F) : ph2;
    ph3 = (ph3 < 0.0f) ? (ph3 + TWO_PI_F) : ph3;

    // 外差合成
    float ph12 = ph1 - ph2;
    if (ph12 < 0.0f) ph12 += TWO_PI_F;
    float ph23 = ph2 - ph3;
    if (ph23 < 0.0f) ph23 += TWO_PI_F;
    float ph123 = ph12 - ph23;
    if (ph123 < 0.0f) ph123 += TWO_PI_F;

    // 展开 ph12
    float expected12 = ph123 * (lambda123 / lambda12);
    int k12 = __float2int_rn((expected12 - ph12) / TWO_PI_F);
    float phi12 = ph12 + TWO_PI_F * k12;

    // 展开最高频率 phi1
    float expected1 = phi12 * (lambda12 / lambda1);
    int k1 = __float2int_rn((expected1 - ph1) / TWO_PI_F);
    float phi_abs = ph1 + TWO_PI_F * k1;

    unwrapped1[idx] = phi_abs;   // 绝对相位 (弧度)
}

// -------------------------------------------------------------------
// 4. 绝对相位 (弧度) 转换为投影仪像素坐标
//    输入: phase (弧度), scale = resolution / (2π * f_high)
//    输出: map (像素坐标)
// -------------------------------------------------------------------
__global__ void phaseToMapKernel(
    const float* phase, float* map,
    float scale, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    int idx = y * cols + x;
    map[idx] = phase[idx] * scale;
}

//自适应设置块和网格，返回是否成功
bool setupAdaptiveLaunch(int width, int height, dim3& threads, dim3& blocks) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 策略1：基于图像尺寸和最大限制选择固定形状
    int targetBlockThreads = 256;   // 经验值
    // 调整使块内线程数不超过 maxThreadsPerBlock
    targetBlockThreads = min(targetBlockThreads, prop.maxThreadsPerBlock);
    // 对齐到 warp 倍数
    targetBlockThreads = (targetBlockThreads / prop.warpSize) * prop.warpSize;

    // 尝试构造二维块尺寸，尽量接近方形
    int bx = sqrt(targetBlockThreads);
    int by = bx;
    while (bx * by > targetBlockThreads) { bx--; by--; }
    // 确保不超过每维最大限制
    bx = min(bx, prop.maxThreadsDim[0]);
    by = min(by, prop.maxThreadsDim[1]);
    // 如果图像宽/高小于块尺寸，可以适当减小块尺寸
    if (bx > width) bx = width;
    if (by > height) by = height;
    // 最终块尺寸
    threads = dim3(bx, by);
    // 网格尺寸
    blocks = dim3(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );
    return true;
}
// -------------------------------------------------------------------
// 5. 宿主端封装函数 (供 C++ 调用)
//    输入: 6组图像 (每组4张灰度图, 顺序: [竖低,竖中,竖高, 横低,横中,横高])
//    输出: mapX, mapY (CV_32FC1)
// -------------------------------------------------------------------
extern "C++" void decodeFringePatterns_GPU(
    const std::vector<std::vector<cv::Mat>>& allFringeImages,  // [6][4] CV_8UC1
    cv::Mat& mapX, cv::Mat& mapY,
    const std::vector<int>& freqs_x,   // 水平条纹频率 (高低顺序)
    const std::vector<int>& freqs_y,   // 竖直条纹频率
    int projWidth, int projHeight)
{
    // 假设 allFringeImages 大小: 6组, 每组4张
    // 前3组为竖直条纹 (索引0,1,2), 后3组为水平条纹 (索引3,4,5)
    int f_num = 3;          // 三频
    int phaseSteps = 4;
    int rows = allFringeImages[0][0].rows;
    int cols = allFringeImages[0][0].cols;

    // 分配 GPU 内存 (使用 GpuMat 简化内存管理)
    std::vector<cv::cuda::GpuMat> gpuImgs(6 * 4);
    for (int g = 0; g < 6; ++g) {
        for (int s = 0; s < 4; ++s) {
            cv::Mat imgFloat;
            allFringeImages[g][s].convertTo(imgFloat, CV_32FC1);
            gpuImgs[g * 4 + s].upload(imgFloat);
        }
    }

    // 定义线程块大小
   /* dim3 threads(32, 32);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);*/
    dim3 threads, blocks;
    setupAdaptiveLaunch(cols, rows, threads, blocks);

    // ---------------- 竖直方向 (X 坐标) ----------------
    // 1. 计算三个包裹相位图
    std::vector<cv::cuda::GpuMat> wrappedY_gpu(f_num);
    for (int i = 0; i < f_num; ++i) {
        wrappedY_gpu[i].create(rows, cols, CV_32FC1);
        computeWrappedPhaseKernel << <blocks, threads >> > (
            (float*)gpuImgs[i * 4 + 0].cudaPtr(),
            (float*)gpuImgs[i * 4 + 1].cudaPtr(),
            (float*)gpuImgs[i * 4 + 2].cudaPtr(),
            (float*)gpuImgs[i * 4 + 3].cudaPtr(),
            (float*)wrappedY_gpu[i].cudaPtr(),
            rows, cols);
    }
    cudaDeviceSynchronize();

    // 2. 准备波长参数 (频率已降序: f0 > f1 > f2)
    int f0_y = freqs_y[0], f1_y = freqs_y[1], f2_y = freqs_y[2];
    float lambda1_y = (float)cols / f0_y;
    float lambda2_y = (float)cols / f1_y;
    float lambda3_y = (float)cols / f2_y;
    float lambda12_y = lambda1_y * lambda2_y / (lambda2_y - lambda1_y);
    float lambda23_y = lambda2_y * lambda3_y / (lambda3_y - lambda2_y);
    float lambda123_y = lambda12_y * lambda23_y / fabs(lambda23_y - lambda12_y);

    cv::cuda::GpuMat absPhaseY_gpu(rows, cols, CV_32FC1);
    unwrapMultiFreqKernel << <blocks, threads >> > (
        (float*)wrappedY_gpu[0].cudaPtr(),
        (float*)wrappedY_gpu[1].cudaPtr(),
        (float*)wrappedY_gpu[2].cudaPtr(),
        (float*)absPhaseY_gpu.cudaPtr(),
        lambda1_y, lambda2_y, lambda3_y,
        lambda12_y, lambda23_y, lambda123_y,
        rows, cols);
    cudaDeviceSynchronize();

    // 3. 转换为投影仪 X 坐标
    cv::cuda::GpuMat mapX_gpu(rows, cols, CV_32FC1);
    float scaleX = projWidth / (TWO_PI_F * f0_y);
    phaseToMapKernel << <blocks, threads >> > (
        (float*)absPhaseY_gpu.cudaPtr(),
        (float*)mapX_gpu.cudaPtr(),
        scaleX, rows, cols);
    cudaDeviceSynchronize();

    // ---------------- 水平方向 (Y 坐标) ----------------
    std::vector<cv::cuda::GpuMat> wrappedX_gpu(f_num);
    for (int i = 0; i < f_num; ++i) {
        int g = i + 3;  // 水平图像组起始索引3
        wrappedX_gpu[i].create(rows, cols, CV_32FC1);
        computeWrappedPhaseKernel << <blocks, threads >> > (
            (float*)gpuImgs[g * 4 + 0].cudaPtr(),
            (float*)gpuImgs[g * 4 + 1].cudaPtr(),
            (float*)gpuImgs[g * 4 + 2].cudaPtr(),
            (float*)gpuImgs[g * 4 + 3].cudaPtr(),
            (float*)wrappedX_gpu[i].cudaPtr(),
            rows, cols);
    }
    cudaDeviceSynchronize();

    int f0_x = freqs_x[0], f1_x = freqs_x[1], f2_x = freqs_x[2];
    float lambda1_x = (float)rows / f0_x;
    float lambda2_x = (float)rows / f1_x;
    float lambda3_x = (float)rows / f2_x;
    float lambda12_x = lambda1_x * lambda2_x / (lambda2_x - lambda1_x);
    float lambda23_x = lambda2_x * lambda3_x / (lambda3_x - lambda2_x);
    float lambda123_x = lambda12_x * lambda23_x / fabs(lambda23_x - lambda12_x);

    cv::cuda::GpuMat absPhaseX_gpu(rows, cols, CV_32FC1);
    unwrapMultiFreqKernel << <blocks, threads >> > (
        (float*)wrappedX_gpu[0].cudaPtr(),
        (float*)wrappedX_gpu[1].cudaPtr(),
        (float*)wrappedX_gpu[2].cudaPtr(),
        (float*)absPhaseX_gpu.cudaPtr(),
        lambda1_x, lambda2_x, lambda3_x,
        lambda12_x, lambda23_x, lambda123_x,
        rows, cols);
    cudaDeviceSynchronize();

    cv::cuda::GpuMat mapY_gpu(rows, cols, CV_32FC1);
    float scaleY = projHeight / (TWO_PI_F * f0_x);
    phaseToMapKernel << <blocks, threads >> > (
        (float*)absPhaseX_gpu.cudaPtr(),
        (float*)mapY_gpu.cudaPtr(),
        scaleY, rows, cols);
    cudaDeviceSynchronize();

    // 下载结果
    mapX_gpu.download(mapX);
    mapY_gpu.download(mapY);

    // 可选: 将结果转换为 CV_64F 以匹配原有数据类型
    // 如果需要 double, 可以在下载后转换: mapX.convertTo(mapX, CV_64F);
}