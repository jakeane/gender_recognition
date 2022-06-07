# Extractors
# June 7th, 2022
# Adrienne Ko (Adrienne.Ko.23@dartmouth.edu)
# Jack Keane (John.F.Keane.22@dartmouth.edu)

# Functions to extract pitch, intonation and resonance from data.

from typing import List, Tuple
from dataclasses import dataclass
import json

import librosa
import numpy as np
import parselmouth as pm
import allosaurus.app
from scipy.io import wavfile
import pyreaper

model = allosaurus.app.read_recognizer("eng2102")

VRANGE = 250

vowel_data = json.load(open("./vowel_formants.json", "r"))
vowels = set(vowel_data["male"]["f1"].keys())
vowel_ranges = {
    v: {
        "f1": {
            "low": vowel_data["male"]["f1"][v] - VRANGE,
            "high": vowel_data["female"]["f1"][v] + VRANGE,
        },
        "f2": {
            "low": vowel_data["male"]["f2"][v] - (VRANGE * 2),
            "high": vowel_data["female"]["f2"][v] + (VRANGE * 2),
        },
    }
    for v in vowels
}
vowel_defaults = {
    v: {
        "f1": (vowel_data["male"]["f1"][v] + vowel_data["female"]["f1"][v]) / 2,
        "f2": (vowel_data["male"]["f2"][v] + vowel_data["female"]["f2"][v]) / 2,
    }
    for v in vowels
}


def get_pitch_intonation(filename, sampling_rate: int) -> Tuple[float, float]:
    y, sr = librosa.load(filename, sr=sampling_rate)
    sound = pm.Sound(y, sr)

    pitches: np.ndarray = sound.to_pitch().selected_array["frequency"]
    pitches = pitches[(pitches != 0) & (pitches < 350)]

    return float(pitches.mean()), float(pitches.std())


def get_pitch_parselmouth(filename, sampling_rate: int) -> Tuple[float, float]:
    y, sr = librosa.load(filename, sr=sampling_rate)
    sound = pm.Sound(y, sr)

    pitches: np.ndarray = sound.to_pitch().selected_array["frequency"]
    pitches = pitches[(pitches != 0) & (pitches < 350)]

    return float(pitches.mean()), float(np.median(pitches))


def get_pitch_librosa(filename, sampling_rate: int) -> Tuple[float, float]:
    y, sr = librosa.load(filename, sr=sampling_rate)

    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=350, sr=sr)
    pitches = f0[vf]

    return float(pitches.mean()), float(np.median(pitches))


def get_pitch_pyreaper(filename) -> Tuple[float, float]:
    sr, y = wavfile.read(filename)

    _, _, _, f0, _ = pyreaper.reaper(y, sr)
    pitches = f0[f0 != -1]

    return float(pitches.mean()), float(np.median(pitches))


@dataclass
class PhoneSegment:
    start: float
    duration: float
    phone: str
    formant1: float = 0
    formant2: float = 0


def get_phones(filename) -> List[PhoneSegment]:
    res: str = model.recognize(filename, timestamp=True)

    segments = res.split("\n")
    phone_times = [(phone, float(time)) for time, _, phone in map(str.split, segments)]
    durations = [
        _clamp(round(phone_times[i][1] - phone_times[i - 1][1], 4))
        for i in range(1, len(phone_times))
    ] + [0.2]
    return [PhoneSegment(t, d, p) for (p, t), d in zip(phone_times, durations)]


def get_formant_diffs(filename, sampling_rate: int) -> Tuple[float, float]:

    f1_diffs = list()
    f2_diffs = list()

    y, sr = librosa.load(filename, sr=sampling_rate)
    phones = get_phones(filename)
    for phone_seg in filter(lambda ps: ps.phone in vowels, phones):
        start = int(sr * phone_seg.start)
        end = int(sr * (phone_seg.start + phone_seg.duration)) + 1

        segment = y[start:end]

        sound = pm.Sound(segment, sr)
        formant = sound.to_formant_burg(None, 5)

        limit = int(1000 * phone_seg.duration)

        f1s = np.fromiter(
            (formant.get_value_at_time(1, i / 1000) for i in range(limit)),
            dtype=np.float32,
        )
        f1s = f1s[~np.isnan(f1s)]
        f2s = np.fromiter(
            (formant.get_value_at_time(2, i / 1000) for i in range(limit)),
            dtype=np.float32,
        )
        f2s = f2s[~np.isnan(f2s)]

        f1, f2 = np.median(f1s), np.median(f2s)
        phone_seg.formant1, phone_seg.formant2 = f1, f2

        phone = phone_seg.phone

        if _are_formants_invalid(phone, f1, f2):
            continue

        f1_default = vowel_defaults[phone]["f1"]
        f2_default = vowel_defaults[phone]["f2"]
        f1_diffs.append(f1 - f1_default)
        f2_diffs.append(f2 - f2_default)

    f1_diff = sum(f1_diffs) / len(f1_diffs)
    f2_diff = sum(f2_diffs) / len(f2_diffs)

    return float(f1_diff), float(f2_diff)


def _are_formants_invalid(phone: str, f1: float, f2: float) -> bool:
    f1_low = vowel_ranges[phone]["f1"]["low"]
    f1_high = vowel_ranges[phone]["f1"]["high"]
    f2_low = vowel_ranges[phone]["f2"]["low"]
    f2_high = vowel_ranges[phone]["f2"]["high"]

    return _outside_range(f1, f1_low, f1_high) or _outside_range(f2, f2_low, f2_high)


def _outside_range(val: float, minv: float, maxv: float) -> bool:
    return val < minv or val > maxv


def _clamp(value: float, minv=0.025, maxv=0.2) -> float:
    return min(maxv, max(minv, value))
