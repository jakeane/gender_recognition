# Gender Recognition Project

Authors: Jack Keane, Adrienne Ko

## Setup
With Python 3.7 or newer
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m allosaurus.bin.download_model -m eng2102
```

## File organization/purpose

 - Python scripts
   - `extract_data.py`: Script to iterate through audio files and extract features
   - `extractors.py`: Feature extraction module
   - `filter_data.py`: Script to determine which files are relevant to the purposed of this project
   - `voxforge.py`: Script to scrape VoxForge website and download audio files
 - Jupyter Notebooks
   - `clean_features.ipynb`: Filter out bad samples and resamples dataset. Also examines final dataset
   - `feedback_system.ipynb`: Interactive feedback system. *key deliverable*
   - `inspect_data.ipynb`: Arbitrary file to view dataset
   - `make_plots.ipynb`: Notebook to make plots for writeup
   - `test_models.ipynb`: Notebook to develop models and explore SHAP *key deliverable*
 - Datasets
   - `audio_features_final.csv`: Dataset used for model development *key deliverable*
   - `audio_features.csv`: Uncleaned dataset, predecessor to `audio_features_final.csv`
   - `audio_files.csv`: Arbitrary dataset of VoxForge files
   - `audio_info_final.csv`: Dataset used for feature extraction
   - `audio_info.csv`: Dataset prior to resampling for class balance, predecessor to `audio_info_final.csv`
   - `audio_pitch_fixed.csv`: Pitch and intonation measurements after algorithm revision, predecessor to `audio_features_final.csv`
   - `audio_pitches.csv`: Exploratory dataset for pitch measurement methods
   - `intonation_important.csv`: Samples where SHAP thought intonation was more imporant that pitch
   - `missed_samples.csv`: Misclassifications made by the machine learning model
   - `outliers.csv`: Collection of files with extremely high pitch, which led to revision for pitch measurement
   - `resonance_important.csv`: Samples where SHAP thought resonance was more imporant that pitch
   - `vowel_formants.json`: Average formant frequencies for vowels by American speakers