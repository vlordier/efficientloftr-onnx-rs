#include <net.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct GpuInstanceGuard
{
    bool active = false;

    ~GpuInstanceGuard()
    {
#if NCNN_VULKAN
        if (active)
            ncnn::destroy_gpu_instance();
#endif
    }
};

namespace
{

void fill_random(ncnn::Mat& m, unsigned int seed)
{
    std::srand(seed);
    float* ptr = (float*)m.data;
    const size_t count = (size_t)m.w * m.h * m.c;
    for (size_t i = 0; i < count; i++)
        ptr[i] = (float)std::rand() / (float)RAND_MAX;
}

void print_shape(const std::string& name, const ncnn::Mat& m)
{
    std::cout << name << ": dims=" << m.dims
              << " w=" << m.w
              << " h=" << m.h
              << " d=" << m.d
              << " c=" << m.c << '\n';
}

double percentile_ms(std::vector<double> values, double fraction)
{
    if (values.empty())
        return 0.0;

    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1, (size_t)(fraction * (values.size() - 1)));
    return values[index];
}

bool run_inference(ncnn::Net& net, const ncnn::Mat& in0, const ncnn::Mat& in1, ncnn::Mat* out0, ncnn::Mat* out1, ncnn::Mat* out2)
{
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(false);

    if (ex.input("in0", in0) != 0)
    {
        std::cerr << "failed to set input in0\n";
        return false;
    }
    if (ex.input("in1", in1) != 0)
    {
        std::cerr << "failed to set input in1\n";
        return false;
    }

    if (ex.extract("out0", *out0) != 0)
    {
        std::cerr << "failed to extract out0\n";
        return false;
    }
    if (ex.extract("out1", *out1) != 0)
    {
        std::cerr << "failed to extract out1\n";
        return false;
    }
    if (ex.extract("out2", *out2) != 0)
    {
        std::cerr << "failed to extract out2\n";
        return false;
    }

    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 5 && argc != 6)
    {
        std::cerr << "usage: " << argv[0] << " <model.param> <model.bin> [warmup runs [vulkan(0|1)]]\n";
        return 2;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const int warmup_runs = argc >= 5 ? std::atoi(argv[3]) : 3;
    const int timed_runs = argc >= 5 ? std::atoi(argv[4]) : 10;

    int use_vulkan = 0;
    if (const char* env_vulkan = std::getenv("NCNN_SMOKE_VULKAN"))
        use_vulkan = std::atoi(env_vulkan) != 0 ? 1 : 0;
    if (argc == 6)
        use_vulkan = std::atoi(argv[5]) != 0 ? 1 : 0;

    if (warmup_runs < 0 || timed_runs <= 0)
    {
        std::cerr << "warmup must be >= 0 and runs must be > 0\n";
        return 2;
    }

    GpuInstanceGuard gpu_instance_guard;
    int gpu_device_index = 0;
    if (const char* env_gpu = std::getenv("NCNN_SMOKE_GPU"))
        gpu_device_index = std::atoi(env_gpu);

#if NCNN_VULKAN
    if (use_vulkan != 0)
    {
        if (ncnn::create_gpu_instance() != 0)
        {
            std::cerr << "failed to create Vulkan instance\n";
            return 2;
        }

        gpu_instance_guard.active = true;

        const int gpu_count = ncnn::get_gpu_count();
        if (gpu_count <= 0)
        {
            std::cerr << "requested vulkan=1 but no Vulkan GPU device found\n";
            return 2;
        }

        if (gpu_device_index < 0 || gpu_device_index >= gpu_count)
        {
            std::cerr << "requested gpu index " << gpu_device_index << " out of range [0," << (gpu_count - 1) << "]\n";
            return 2;
        }
    }
#else
    if (use_vulkan != 0)
    {
        std::cerr << "requested vulkan=1 but ncnn was built without Vulkan support\n";
        return 2;
    }
#endif

    const int ret = [&]() -> int {
        ncnn::Net net;
        net.opt.use_vulkan_compute = use_vulkan != 0;

#if NCNN_VULKAN
        if (net.opt.use_vulkan_compute)
            net.set_vulkan_device(gpu_device_index);
#endif

        if (net.load_param(param_path) != 0)
        {
            std::cerr << "failed to load param: " << param_path << '\n';
            return 1;
        }

        if (net.load_model(bin_path) != 0)
        {
            std::cerr << "failed to load bin: " << bin_path << '\n';
            return 1;
        }

        ncnn::Mat in0(640, 480, 1);
        ncnn::Mat in1(640, 480, 1);
        fill_random(in0, 42u);
        fill_random(in1, 1337u);

        ncnn::Mat out0;
        ncnn::Mat out1;
        ncnn::Mat out2;

        if (!run_inference(net, in0, in1, &out0, &out1, &out2))
            return 1;

        for (int i = 0; i < warmup_runs; i++)
        {
            if (!run_inference(net, in0, in1, &out0, &out1, &out2))
                return 1;
        }

        std::vector<double> timings_ms;
        timings_ms.reserve((size_t)timed_runs);
        for (int i = 0; i < timed_runs; i++)
        {
            const auto start = std::chrono::steady_clock::now();
            if (!run_inference(net, in0, in1, &out0, &out1, &out2))
                return 1;
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
            timings_ms.push_back(elapsed.count());
        }

        print_shape("out0", out0);
        print_shape("out1", out1);
        print_shape("out2", out2);

        double total_ms = 0.0;
        double min_ms = timings_ms.front();
        double max_ms = timings_ms.front();
        for (double value : timings_ms)
        {
            total_ms += value;
            min_ms = std::min(min_ms, value);
            max_ms = std::max(max_ms, value);
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "benchmark_config: vulkan=" << (net.opt.use_vulkan_compute ? 1 : 0)
                  << " "
                  << "benchmark: warmup=" << warmup_runs
                  << " runs=" << timed_runs
                  << " mean_ms=" << (total_ms / timings_ms.size())
                  << " p50_ms=" << percentile_ms(timings_ms, 0.50)
                  << " p90_ms=" << percentile_ms(timings_ms, 0.90)
                  << " min_ms=" << min_ms
                  << " max_ms=" << max_ms << '\n';

        std::cout << "ncnn smoke inference ok\n";
        return 0;
    }();

    return ret;
}
