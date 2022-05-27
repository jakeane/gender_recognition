import os
import re

import pandas as pd
from tqdm import tqdm


def mcgill():
    data = list()
    for directory in os.listdir("./audio_data/mcgill"):
        if directory.startswith("F") or directory.startswith("M"):
            is_male = directory.startswith("M")
            for f in os.listdir(f"./audio_data/mcgill/{directory}"):
                if f.endswith(".wav"):
                    data.append(
                        {
                            "filename": f"./audio_data/mcgill/{directory}/{f}",
                            "gender": "male" if is_male else "female",
                            "sampling_rate": 48000,
                            "dialect": "Canadian English",
                            "source": "McGill",
                            "mic_type": "stereo mic",
                        }
                    )

    # df = pd.DataFrame.from_records(data)
    # df.to_csv("./audio_files.csv")
    return data


def voxforge():
    data = list()
    for d in tqdm(os.listdir("./audio_data/voxforge")):
        with open(f"./audio_data/voxforge/{d}/etc/README") as f:
            text = f.read()

        gender = re.search(r"Gender: ([\w ]+)", text)
        age_range = re.search(r"Age Range: ([\w ]+)", text)
        language = re.search(r"Language: ([\w ]+)", text)
        dialect = re.search(r"Pronunciation dialect: ([\w ]+)", text)
        sampling_rate = re.search(r"Sampling Rate: ([\w ]+)", text)
        file_type = re.search(r"File type: ([\w ]+)", text)
        mic_type = re.search(r"Microphone type: ([\w ]+)", text)
        recording_software = re.search(r"Audio Recording Software: ([\w ]+)", text)

        info = {
            "key": d,
            "gender": gender.group(1) if gender else "N/A",
            "age_range": age_range.group(1) if age_range else "N/A",
            "language": language.group(1) if language else "N/A",
            "dialect": dialect.group(1) if dialect else "N/A",
            "sampling_rate": sampling_rate.group(1) if sampling_rate else "N/A",
            "file_type": file_type.group(1) if file_type else "N/A",
            "mic_type": mic_type.group(1) if mic_type else "N/A",
            "recording_software": recording_software.group(1)
            if recording_software
            else "N/A",
        }

        if info["gender"].lower() not in ["male", "female"]:
            continue
        if info["age_range"].lower() != "adult":
            continue
        if info["dialect"].lower().strip() not in [
            "american english",
            "canadian english",
            "general american english",
            "america english",
            "general american",
        ]:
            continue
        if info["file_type"].lower() != "wav":
            continue
        if info["mic_type"].lower() in [
            "unknown",
            "please select",
            "other",
            "webcam mic",
        ]:
            continue

        for i, audio in enumerate(os.listdir(f"./audio_data/voxforge/{d}/wav")):
            # if info["gender"].lower() == "male" and i > 1:
            #     break
            data.append(
                {
                    "filename": f"./audio_data/voxforge/{d}/wav/{audio}",
                    "gender": info["gender"].lower(),
                    "sampling_rate": 48000,
                    "dialect": info["dialect"],
                    "source": "VoxForge",
                    "mic_type": info["mic_type"].lower(),
                }
            )

    return data


def main():
    mg_data = mcgill()
    vf_data = voxforge()
    df = pd.DataFrame.from_records([*mg_data, *vf_data])
    df.to_csv("./audio_info.csv")


if __name__ == "__main__":
    main()
