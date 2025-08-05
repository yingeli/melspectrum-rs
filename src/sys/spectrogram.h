#pragma once

#include "spectrogram/src/sys/features.h"

#include "rust/cxx.h"

#include <torch/torch.h>
#include <filesystem>
#include <span>

enum class DeviceType: std::uint8_t;

class LogMelSpectrogram {
    public:
        LogMelSpectrogram(
            const std::filesystem::path& mel_filter_path,
            const size_t n_mels,
            const size_t n_fft,
            const size_t hop_length,
            const DeviceType device
        );

        inline std::unique_ptr<Features> extract(
            const rust::Slice<const float> samples,
            const size_t padding = 0
        ) const {
            return std::make_unique<Features>(
                extract(
                    std::span<const float>(samples.data(), samples.size()),
                    padding
                )
            );
        }

        inline std::unique_ptr<Features> extract_multi(
            const rust::Slice<const float> first,
            const rust::Slice<const float> second,
            const size_t padding = 0
        ) const {
            return std::make_unique<Features>(
                extract(
                    std::span<const float>(first.data(), first.size()),
                    std::span<const float>(second.data(), second.size()),
                    padding
                )
            );
        }


        inline std::unique_ptr<Features> empty() const {
            // auto tensor = torch::empty({n_mels(), 0}, torch::TensorOptions().dtype(torch::kFloat32).device(filters_.device()));
            auto tensor = torch::empty({0, n_mels()}, torch::TensorOptions().dtype(torch::kFloat16).device(filters_.device()));
            return std::make_unique<Features>(tensor);
        }

        size_t n_mels() const {
            return filters_.size(0);
        }

        size_t n_fft() const {
            return n_fft_;
        }

        size_t hop_length() const {
            return hop_length_;
        }

    private:
        torch::Tensor compute(
            const torch::Tensor& samples
        ) const;

        torch::Tensor extract(
            const std::span<const float> samples,
            const size_t padding = 0
        ) const;

        torch::Tensor extract(
            const std::span<const float> first,
            const std::span<const float> second,
            const size_t padding = 0
        ) const;

        torch::Tensor filters_;
        torch::Tensor window_;
        size_t n_fft_;
        size_t hop_length_;
};

inline std::unique_ptr<LogMelSpectrogram> log_mel_spectrogram(
    const rust::Str mel_filter_path,
    const size_t n_mels,
    const size_t n_fft,
    const size_t hop_length,
    const DeviceType device
) {
    auto path = std::filesystem::path(static_cast<std::string>(mel_filter_path));
    return std::make_unique<LogMelSpectrogram>(path, n_mels, n_fft, hop_length, device);
}