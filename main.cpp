#include <iostream>
#include <opencv2/opencv.hpp>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/providers/cuda/cuda_provider_factory.h>
#include <typeinfo>

using namespace std;
using namespace cv;
using namespace Ort;

int main()
{
    Mat input_image = imread("/home/lynn/dataset/valid/ok/img_NG_SpringSegment_20220905_110434_286.png");
    imshow("img", input_image);
    resize(input_image, input_image, Size(224, 224));
    int rows = input_image.rows;
    int cols = input_image.cols;
    int channels = 3;
    vector<float> mean = {0.485f, 0.456f, 0.406f};
    vector<float> var = {0.229f, 0.224f, 0.225f};
    vector<float> input_vector(224 * 224 * 3);
    for (auto c = 0; c < channels; c++)
        for (auto i = 0; i < rows; i++)
            for (auto j = 0; j < cols; j++)
            {
                input_vector[c * rows * cols + i * cols + j] = float(input_image.at<Vec3b>(j, i)[2 - c]) / 255.f;
                input_vector[c * rows * cols + i * cols + j] = (input_vector[c * rows * cols + i * cols + j] - mean[c]) / var[c];
            }
    // 环境
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "class");
    Ort::SessionOptions session_options;
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::Session session(env, "/home/lynn/best.onnx", session_options);

    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char *> input_node_names(num_input_nodes);
    for (auto i = 0; i < num_input_nodes; i++)
    {
        input_node_names[i] = session.GetInputName(i, allocator);
    }

    size_t input_tensor_size = 224 * 224 * 3; // simplify ... using known dim values to calculate size
                                              // use OrtGetTensorShapeElementCount() to get official size!

    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char *> output_node_names = {session.GetOutputName(0, allocator)};

    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto input_node_dims = tensor_info.GetShape();
    input_node_dims[0] = 1;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    while (1)
    {
        int64 start = getTickCount();
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_vector.data(), input_tensor_size, input_node_dims.data(), 4);

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        float *floatarr = output_tensors.front().GetTensorMutableData<float>();
        int64 end = getTickCount();
        cout << (end - start) * 1000 / getTickFrequency() << endl;
        for (int i = 0; i < 2; i++)
        {
            cout << floatarr[i] << endl;
        }
    }
}