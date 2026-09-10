"""
vad_test.py
-----------
Stage 1 of the pipeline: Voice Activity Detection (VAD).

Takes a .wav file, splits it into short frames, and classifies each
frame as speech or silence using WebRTC's VAD (a lightweight, fast,
CPU-only detector -- runs fine on a laptop and later on the Pi 5).

This is the very first stage of the pipeline described in the project
report:
    Mic -> VAD (trim silence) -> Segmentation -> Embedding extraction
    -> Clustering -> Diarization output -> Identification

Usage:
    python3 vad_test.py path/to/audio.wav

Requires:
    pip install webrtcvad soundfile numpy
"""

import sys
import wave
import contextlib
import webrtcvad
import numpy as np


def read_wave(path):
    """Read a WAV file and return (pcm_bytes, sample_rate).

    WebRTC VAD requires mono, 16-bit PCM, at 8000/16000/32000/48000 Hz.
    """
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())

    if num_channels != 1:
        raise ValueError(
            f"Expected mono audio, got {num_channels} channels. "
            "Convert with: ffmpeg -i input.wav -ac 1 -ar 16000 output.wav"
        )
    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM, got {sample_width * 8}-bit. "
            "Convert with: ffmpeg -i input.wav -ac 1 -ar 16000 -sample_fmt s16 output.wav"
        )
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError(
            f"Sample rate {sample_rate} not supported by webrtcvad. "
            "Use 8000, 16000, 32000, or 48000 Hz. "
            "Convert with: ffmpeg -i input.wav -ar 16000 output.wav"
        )

    return pcm_data, sample_rate


def frame_generator(frame_duration_ms, pcm_data, sample_rate):
    """Yield successive frames of frame_duration_ms from pcm_data."""
    bytes_per_sample = 2  # 16-bit
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0)) * bytes_per_sample
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0

    while offset + frame_size <= len(pcm_data):
        yield pcm_data[offset:offset + frame_size], timestamp
        timestamp += duration
        offset += frame_size


def run_vad(path, aggressiveness=2, frame_duration_ms=30):
    """
    Run VAD over the given wav file and print speech/silence segments.

    aggressiveness: 0-3. Higher = more aggressive about filtering out
                    non-speech (fewer false positives, but may clip
                    quiet speech). 2 is a reasonable default.
    frame_duration_ms: must be 10, 20, or 30 (webrtcvad requirement).
    """
    pcm_data, sample_rate = read_wave(path)
    vad = webrtcvad.Vad(aggressiveness)

    frames = list(frame_generator(frame_duration_ms, pcm_data, sample_rate))
    if not frames:
        print("No frames produced -- is the file long enough / valid?")
        return []

    segments = []  # list of (start_time, end_time, is_speech)
    current_label = None
    seg_start = 0.0

    for frame_bytes, timestamp in frames:
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if current_label is None:
            current_label = is_speech
            seg_start = timestamp
        elif is_speech != current_label:
            segments.append((seg_start, timestamp, current_label))
            current_label = is_speech
            seg_start = timestamp

    # close out the final segment
    last_timestamp = frames[-1][1] + (frame_duration_ms / 1000.0)
    segments.append((seg_start, last_timestamp, current_label))

    return segments


def print_report(segments):
    if not segments:
        return

    total_duration = segments[-1][1]
    speech_time = sum(end - start for start, end, is_speech in segments if is_speech)
    silence_time = total_duration - speech_time

    print(f"{'Start':>8}  {'End':>8}  {'Label'}")
    print("-" * 32)
    for start, end, is_speech in segments:
        label = "SPEECH" if is_speech else "silence"
        print(f"{start:8.2f}  {end:8.2f}  {label}")

    print("-" * 32)
    print(f"Total duration : {total_duration:6.2f}s")
    print(f"Speech time    : {speech_time:6.2f}s ({100 * speech_time / total_duration:5.1f}%)")
    print(f"Silence time   : {silence_time:6.2f}s ({100 * silence_time / total_duration:5.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 vad_test.py path/to/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    segments = run_vad(wav_path)
    print_report(segments)
