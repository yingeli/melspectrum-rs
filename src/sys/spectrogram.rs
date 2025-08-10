use super::features::{self, Features};

use cxx::UniquePtr;

use anyhow::{Result, anyhow};
use std::path::Path;

pub use ffi::DeviceType;

#[cxx::bridge]
pub(crate) mod ffi {
    #[derive(Copy, Clone, Debug)]
    #[repr(u8)]
    enum DeviceType {
        Cpu,
        Cuda,
    }

    unsafe extern "C++" {
        type Features = super::features::ffi::Features;

        include!("spectrogram/src/sys/spectrogram.h");

        type LogMelSpectrogram;

        fn log_mel_spectrogram(
            mel_filter_path: &str,
            n_mels: usize,
            n_fft: usize,
            hop_length: usize,
            device: DeviceType,
        ) -> Result<UniquePtr<LogMelSpectrogram>>;

        fn extract(
            self: &LogMelSpectrogram,
            samples: &[f32],
            padding: usize,
        ) -> Result<UniquePtr<Features>>;

        fn extract_multi(
            self: &LogMelSpectrogram,
            first: &[f32],
            second: &[f32],
            padding: usize,
        ) -> Result<UniquePtr<Features>>;

        fn empty(self: &LogMelSpectrogram) -> UniquePtr<Features>;

        fn n_fft(self: &LogMelSpectrogram) -> usize;

        fn hop_length(self: &LogMelSpectrogram) -> usize;
    }
}

pub struct LogMelSpectrogram {
    ptr: UniquePtr<ffi::LogMelSpectrogram>,
}

impl LogMelSpectrogram {
    pub fn open(
        mel_filter_path: impl AsRef<Path>,
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        device: DeviceType,
    ) -> Result<Self> {
        let path = mel_filter_path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow!("failed to convert mel filter path to str"))?;
        let ptr = ffi::log_mel_spectrogram(path, n_mels, n_fft, hop_length, device)
            .map_err(|e| anyhow!("failed to create log mel spectrogram: {}", e))?;

        Ok(Self { ptr })
    }

    pub fn n_fft(&self) -> usize {
        self.ptr.n_fft()
    }

    pub fn hop_length(&self) -> usize {
        self.ptr.hop_length()
    }

    pub fn n_left_overlap_frames(&self) -> usize {
        (self.n_fft() / 2 + self.hop_length() - 1) / self.hop_length()
    }

    pub fn n_left_overlap_samples(&self) -> usize {
        self.n_left_overlap_frames() * self.hop_length()
    }

    pub fn n_right_overlap_samples(&self) -> usize {
        self.n_fft() / 2 - self.hop_length()
    }

    pub fn extract(&self, samples: &[f32], padding: usize) -> Result<Features> {
        let ptr = self
            .ptr
            .extract(samples, padding)
            .map_err(|e| anyhow!("failed to extract log mel spectrogram: {}", e))?;

        Ok(ptr.into())      
    }

    pub fn extract_multi(&self, first: &[f32], second: &[f32], padding: usize) -> Result<Features> {
        let ptr = self
            .ptr
            .extract_multi(first, second, padding)
            .map_err(|e| anyhow!("failed to extract log mel spectrogram: {}", e))?;

        Ok(ptr.into())
    }

    pub fn empty(&self) -> Features {
        self.ptr.empty().into()
    }
}

unsafe impl Send for LogMelSpectrogram {}
unsafe impl Sync for LogMelSpectrogram {}

#[cfg(test)]
mod tests {
    use super::{LogMelSpectrogram, DeviceType};

    #[test]
    fn test_open() {
        let mel_filter_path = "../../assets/mel_filter.npz";
        let n_mels = 128;
        let n_fft = 400;
        let hop_length = 160;
        let device = DeviceType::Cpu;

        let spectrogram = LogMelSpectrogram::open(mel_filter_path, n_mels, n_fft, hop_length, device);
        assert!(spectrogram.is_ok());

        let spectrogram = spectrogram.unwrap();
        assert_eq!(spectrogram.n_fft(), n_fft);
        assert_eq!(spectrogram.hop_length(), hop_length);
        assert_eq!(spectrogram.n_left_overlap_frames(), 2);
        assert_eq!(spectrogram.n_left_overlap_samples(), 320);
        assert_eq!(spectrogram.n_right_overlap_samples(), 40);
    }
}