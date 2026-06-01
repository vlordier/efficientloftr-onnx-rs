#include <net.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

void fill_random(ncnn::Mat& m, unsigned int seed)
{
    std::srand(seed);
    float* ptr = (float*)m.data;
    const size_t count = (size_t)m.w * m.h * m.c;
    for (size_t i = 0; i < count; i++)
    {
        ptr[i] = (float)std::rand() / (float)RAND_MAX;
    }
}

void print_shape(const std::string& name, const ncnn::Mat& m)
{
    std::cout << name << ": dims=" << m.dims
              << " w=" << m.w
              << " h=" << m.h
              << " d=" << m.d
              << " c=" << m.c << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: " << argv[0] << " <model.param> <model.bin>\n";
        return 2;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;

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

    // Model expects grayscale BCHW with B=1, C=1, H=480, W=640.
    ncnn::Mat in0(640, 480, 1);
    ncnn::Mat in1(640, 480, 1);
    fill_random(in0, 42u);
    fill_random(in1, 1337u);

    ncnn::Extractor ex = net.create_extractor();

    if (ex.input("in0", in0) != 0)
    {
        std::cerr << "failed to set input in0\n";
        return 1;
    }
    if (ex.input("in1", in1) != 0)
    {
        std::cerr << "failed to set input in1\n";
        return 1;
    }

    ncnn::Mat out0;
    ncnn::Mat out1;
    ncnn::Mat out2;

    if (ex.extract("out0", out0) != 0)
    {
        std::cerr << "failed to extract out0\n";
        return 1;
    }
    if (ex.extract("out1", out1) != 0)
    {
        std::cerr << "failed to extract out1\n";
        return 1;
    }
    if (ex.extract("out2", out2) != 0)
    {
        std::cerr << "failed to extract out2\n";
        return 1;
    }

    print_shape("out0", out0);
    print_shape("out1", out1);
    print_shape("out2", out2);

    std::cout << "ncnn smoke inference ok\n";
    return 0;
}
