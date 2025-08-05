#include "spectrogram/src/sys/spectrogram.rs.h"

#include "cnpy/cnpy.h"

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <string>
#include <vector>
#include <filesystem>

LogMelSpectrogram::LogMelSpectrogram(
    const std::filesystem::path& mel_filter_path,
    const size_t n_mels,
    const size_t n_fft,
    const size_t hop_length,
    const DeviceType device
) : n_fft_(n_fft),
    hop_length_(hop_length) {
    // Load the npz file
    cnpy::npz_t file = cnpy::npz_load(mel_filter_path);

    // Load the mel filter array
    std::string mel_name = std::string("mel_") + std::to_string(n_mels);
    cnpy::NpyArray mel_array = file[mel_name];
    std::vector<int64_t> shape(mel_array.shape.begin(), mel_array.shape.end());

    auto dev = (device == DeviceType::Cuda) ? torch::kCUDA : torch::kCPU;
    filters_ = torch::from_blob(mel_array.data<void>(), shape, torch::kFloat32).to(dev);
    window_ = torch::hann_window(n_fft).to(dev);
}

torch::Tensor LogMelSpectrogram::extract(
    const std::span<const float> samples,
    const size_t padding
) const {
    auto device = filters_.device();

    torch::Tensor input = torch::from_blob(
        const_cast<float*>(samples.data()), 
        {static_cast<long>(samples.size())}, 
        torch::kFloat32
    );
    
    if (!device.is_cpu()) {
        input = input.to(device);
    }

    if (padding > 0) {
        input = torch::nn::functional::pad(
            input,
            torch::nn::functional::PadFuncOptions({0, padding})
        );
    }

    return compute(input);
}

torch::Tensor LogMelSpectrogram::extract(
    const std::span<const float> first,
    const std::span<const float> second,
    const size_t padding
) const {
    auto device = filters_.device();

    const auto total_size = first.size() + second.size();
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device);

    auto input = torch::empty({static_cast<long>(total_size + padding)}, options);

    auto tensor1 = torch::from_blob(
        const_cast<float*>(first.data()),
        {static_cast<long>(first.size())},
        torch::kFloat32
    );

    auto tensor2 = torch::from_blob(
        const_cast<float*>(second.data()),
        {static_cast<long>(second.size())},
        torch::kFloat32
    );

    input.slice(0, 0, first.size()).copy_(tensor1, true);
    input.slice(0, first.size(), total_size).copy_(tensor2, true);
    if (padding > 0) {
        input.slice(0, total_size).zero_();
    }

    return compute(input);
}

torch::Tensor LogMelSpectrogram::compute(
    const torch::Tensor& samples
) const {
    torch::Tensor stft = torch::stft(samples, 
        n_fft_, 
        hop_length_, 
        n_fft_, 
        window_,
        true, // center
        "reflect", // pad_mode
        false, // normalized
        true, // onesided
        true // return_complex
    );

    auto magnitudes = stft.slice(-1, 0, stft.size(-1) - 1).abs().pow(2);

    torch::Tensor mel_spec = torch::matmul(filters_, magnitudes);

    auto log_spec = torch::clamp_min(mel_spec, 1e-10f).log10_();

    torch::maximum_out(log_spec, log_spec, log_spec.max() - 8.0f);
    
    return log_spec.add_(4.0f)
        .div_(4.0f)
        .to(torch::kFloat16)
        .transpose_(0, 1);
}