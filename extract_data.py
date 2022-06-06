from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from extractors import get_formant_diffs, get_pitch_intonation


def main():

    df = pd.read_csv("./audio_info_final.csv", index_col=0)
    df_list = df.to_dict("records")

    with ThreadPoolExecutor(max_workers=8) as executor:
        # data = list(tqdm(executor.map(extract_features, df_list), total=len(df_list)))
        data = list(tqdm(executor.map(extract_pitch, df_list), total=len(df_list)))

    features_df = pd.DataFrame.from_records(data)
    # features_df.to_csv("./audio_features.csv")
    features_df.to_csv("./audio_pitch_fixed.csv")


def extract_pitch(row):
    try:
        pitch, intonation = get_pitch_intonation(row["filename"], 48000)
    except Exception as e:
        print("error with", row["filename"], e)
        pitch, intonation = None, None

    return {**row, "pitch": pitch, "intonation": intonation}


def extract_features(row):

    try:
        pitch, intonation = get_pitch_intonation(row["filename"], 48000)
    except Exception:
        pitch, intonation = None, None
    try:
        f1_diff, f2_diff = get_formant_diffs(row["filename"], 48000)
    except Exception:
        f1_diff, f2_diff = None, None

    return {
        **row,
        "pitch": pitch,
        "intonation": intonation,
        "f1_diff": f1_diff,
        "f2_diff": f2_diff,
    }


if __name__ == "__main__":
    main()
