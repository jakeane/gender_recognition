# Scrape VoxForge Data
# June 7th, 2022
# Adrienne Ko (Adrienne.Ko.23@dartmouth.edu)
# Jack Keane (John.F.Keane.22@dartmouth.edu)

# Scrapes the VoxForge website for voice samples.

import tarfile
import io
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def main():
    url = "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/"
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    files = [
        f'{url}/{link["href"]}'
        for link in soup.find_all("a")
        if link["href"].endswith(".tgz")
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        for f in tqdm(files):
            executor.submit(get_data, f)


def get_data(url):
    data = requests.get(url)
    with tarfile.open(fileobj=io.BytesIO(data.content)) as tf:
        tf.extractall("./audio_data/voxforge")


if __name__ == "__main__":
    main()
