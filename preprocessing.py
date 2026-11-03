import librosa
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.effects import compress_dynamic_range, normalize
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path("../Preprocessed")
BASE_DIR.mkdir(parents=True, exist_ok=True)

def normalize_volume(file_path, file_name):
    # Load audio
    audio = AudioSegment.from_wav(file_path)

    # Normalize volume
    normalized_audio = normalize(audio)

    # Save
    output_path = BASE_DIR / f"normalized_{file_name}.wav"
    normalized_audio.export(output_path, format="wav")

    # ---------- Plot ----------
    original_audio, sr = librosa.load(file_path, sr=None)
    normalized_waveform, _ = librosa.load(output_path, sr=sr)

    # plt.figure(figsize=(12, 6))

    # plt.subplot(2, 1, 1)
    # librosa.display.waveshow(original_audio, sr=sr)
    # plt.title("Original Audio")

    # plt.subplot(2, 1, 2)
    # librosa.display.waveshow(normalized_waveform, sr=sr)
    # plt.title("Normalized Audio")

    # plt.tight_layout()
    # plt.show()

    return output_path


def noise_reduction(file_path, file_name):
    audio, sr = librosa.load(file_path, sr=None)
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    output_path = BASE_DIR / f"pre{file_name}.wav"
    sf.write(output_path, reduced_noise, sr)

    # plt.figure(figsize=(12, 6))

    # plt.subplot(2, 1, 1)
    # librosa.display.waveshow(audio, sr=sr)
    # plt.title("Original Audio")

    # plt.subplot(2, 1, 2)
    # librosa.display.waveshow(reduced_noise, sr=sr)
    # plt.title("Noise Reduced Audio")

    # plt.tight_layout()
    # plt.show()


    return output_path


def silence_removal(original_file_path, file_path, file_name):
    # Load with pydub (milliseconds)
    audio = AudioSegment.from_wav(file_path)

    # Detect speech segments
    intervals = detect_nonsilent(
        audio,
        min_silence_len=1200,   # Remove only silence longer than 1 second
        silence_thresh=-60      # Adjust if necessary
    )

    keep_silence = 500  # Keep 500 ms before and after speech

    processed = AudioSegment.empty()

    for start, end in intervals:
        start = max(0, start - keep_silence)
        end = min(len(audio), end + keep_silence)
        processed += audio[start:end]

    output_path = BASE_DIR / f"{file_name}.wav"
    processed.export(output_path, format="wav")

    # ---------- Plot ----------
    original_audio, sr = librosa.load(original_file_path, sr=None)
    processed_audio, _ = librosa.load(output_path, sr=sr)

    # plt.figure(figsize=(12, 6))

    # plt.subplot(2, 1, 1)
    # librosa.display.waveshow(original_audio, sr=sr)
    # plt.title("Original Audio")

    # plt.subplot(2, 1, 2)
    # librosa.display.waveshow(processed_audio, sr=sr)
    # plt.title("Preprocessed Audio")

    # plt.tight_layout()
    # plt.show()

    return output_path

def compress_volume(original_file_path, file_path, file_name):
    # Load audio
    audio = AudioSegment.from_wav(file_path)

    # Compress dynamic range
    compressed_audio = compress_dynamic_range(
        audio,
        threshold=-20.0,   # dBFS above which compression starts
        ratio=4.0,         # Compression ratio
        attack=5,          # ms
        release=50         # ms
    )

    # Optional: Normalize after compression
    compressed_audio = normalize(compressed_audio)

    # Save
    output_path = BASE_DIR / f"compressed_{file_name}.wav"
    compressed_audio.export(output_path, format="wav")

    # ---------- Plot ----------
    original_audio, sr = librosa.load(file_path, sr=None)
    compressed_waveform, _ = librosa.load(output_path, sr=sr)

    # plt.figure(figsize=(12, 6))

    # plt.subplot(2, 1, 1)
    # librosa.display.waveshow(original_audio, sr=sr)
    # plt.title("Original Audio")

    # plt.subplot(2, 1, 2)
    # librosa.display.waveshow(compressed_waveform, sr=sr)
    # plt.title("Compressed + Normalized Audio")

    # plt.tight_layout()
    # plt.show()

    return output_path


# print(noise_reduction("../Dataset/53.wav", "1"))
# print(silence_removal("../Dataset/53.wav", "1"))
# print(normalize_volume("../Dataset/53.wav", "1"))
# print(compress_volume("../Dataset/53.wav", "1"))


import time
import os

def preprocess_audio(file_path, file_name):

    total_start = time.perf_counter()

    # Step 1: Noise Reduction
    # start = time.perf_counter()
    # noise_reduced_path = noise_reduction(file_path, file_name)
    # noise_time = time.perf_counter() - start

    # Step 2: Compression
    start = time.perf_counter()
    compressed_path = compress_volume(file_path, file_path, file_name)
    compression_time = time.perf_counter() - start

    # Step 3: Silence Removal
    start = time.perf_counter()
    final_path = silence_removal(file_path, compressed_path, file_name)
    silence_time = time.perf_counter() - start

    total_time = time.perf_counter() - total_start

    # print(f"Noise Reduction: {noise_time:.3f} s")
    print(f"Compression:     {compression_time:.3f} s")
    print(f"Silence Removal: {silence_time:.3f} s")
    print(f"Total:           {total_time:.3f} s")

    # if os.path.exists(noise_reduced_path):
    #     os.remove(noise_reduced_path)

    if os.path.exists(compressed_path):
        os.remove(compressed_path)

    return final_path


# print(preprocess_audio("../Dataset/1.wav", "1"))
