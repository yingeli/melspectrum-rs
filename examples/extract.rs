use spectrogram::{DeviceType, LogMelSpectrogram};
use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    let mel_filter_path = "assets/mel_filters.npz";
    let n_mels = 128;
    let n_fft = 400;
    let hop_length = 160;
    let device = DeviceType::Cpu; // Change to DeviceType::Cuda if using CUDA

    let spectrogram = LogMelSpectrogram::open(mel_filter_path, n_mels, n_fft, hop_length, device)?;
    println!("Log Mel Spectrogram created successfully!");

    // Measure time for 100 runs
    let num_iterations = 1000;

    // Prepare audio data (30 seconds at 16kHz)
    let samples = vec![1.0; 16000 * 30];
    
    // Warm-up run
    let _ = spectrogram.extract(&samples, 0)?;

    let start = Instant::now();
    
    for _i in 0..num_iterations {
        let _features = spectrogram.extract(&samples, 0)?;
    }
    
    let duration = start.elapsed();
    
    // Calculate and display statistics
    println!("\n=== Performance Statistics ===");
    println!("Total iterations: {}", num_iterations);
    println!("Total time: {:.3} seconds", duration.as_secs_f64());
    println!("Average time per iteration: {:.3} ms", duration.as_secs_f64() * 1000.0 / num_iterations as f64);
    println!("Throughput: {:.2} iterations/second", num_iterations as f64 / duration.as_secs_f64());
    println!("Audio processed: {:.1} seconds per iteration", 30.0);
    println!("Real-time factor: {:.2}x", (30.0 * num_iterations as f64) / duration.as_secs_f64());


        // Prepare audio data (30 seconds at 16kHz)
    let samples = vec![1.0; 16000 * 20];
    
    // Warm-up run
    let _ = spectrogram.extract(&samples, 16000 * 10)?;

    let start = Instant::now();
    
    for _i in 0..num_iterations {
        let _features = spectrogram.extract(&samples, 16000 * 10)?;
    }
    
    let duration = start.elapsed();
    
    // Calculate and display statistics
    println!("\n=== Performance Statistics ===");
    println!("Total iterations: {}", num_iterations);
    println!("Total time: {:.3} seconds", duration.as_secs_f64());
    println!("Average time per iteration: {:.3} ms", duration.as_secs_f64() * 1000.0 / num_iterations as f64);
    println!("Throughput: {:.2} iterations/second", num_iterations as f64 / duration.as_secs_f64());
    println!("Audio processed: {:.1} seconds per iteration", 30.0);
    println!("Real-time factor: {:.2}x", (30.0 * num_iterations as f64) / duration.as_secs_f64());


    // Prepare audio data (30 seconds at 16kHz)
    let first = vec![1.0; 16000 * 20];
    let second = vec![1.0; 16000 * 10];
    
    // Warm-up run
    let _ = spectrogram.extract_multi(&first, &second, 0)?;

    let start = Instant::now();
    
    for _i in 0..num_iterations {
        let _features = spectrogram.extract_multi(&first, &second, 0)?;
    }
    
    let duration = start.elapsed();
    
    // Calculate and display statistics
    println!("\n=== Performance Statistics ===");
    println!("Total iterations: {}", num_iterations);
    println!("Total time: {:.3} seconds", duration.as_secs_f64());
    println!("Average time per iteration: {:.3} ms", duration.as_secs_f64() * 1000.0 / num_iterations as f64);
    println!("Throughput: {:.2} iterations/second", num_iterations as f64 / duration.as_secs_f64());
    println!("Audio processed: {:.1} seconds per iteration", 30.0);
    println!("Real-time factor: {:.2}x", (30.0 * num_iterations as f64) / duration.as_secs_f64());

    Ok(())
}