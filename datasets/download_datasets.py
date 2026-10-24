"""
Download the public datasets used in Section 8 of the paper into this folder.

    python datasets/download_datasets.py            # SONG and RNA
    python datasets/download_datasets.py song       # just one

KDD (KDD Cup 2004 protein homology) needs a free registration, so it can't be
downloaded by script. Get data_kddcup04.tar.gz from
https://osmot.cs.cornell.edu/kddcup/datasets.html, extract it, and copy
bio_train.dat into this folder.

CSN (Community Seismic Network accelerometer data) is not publicly available.
"""

import os
import shutil
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

SONG_URL = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
RNA_URLS = [
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",    # 59,535 rows
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t",  # 271,617 rows
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r",  # 157,413 rows
]


def fetch(url, dest):
    if os.path.exists(dest):
        print(f"  {os.path.basename(dest)} already present")
        return
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)


def get_song():
    out = os.path.join(HERE, "YearPredictionMSD.txt")
    if os.path.exists(out):
        print("SONG: already present")
        return
    print("SONG (UCI YearPredictionMSD, 515,345 x 90, ~200 MB zipped)")
    zpath = os.path.join(HERE, "yearpredictionmsd.zip")
    fetch(SONG_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(HERE)
    os.remove(zpath)


def get_rna():
    out = os.path.join(HERE, "cod-rna.all")
    if os.path.exists(out):
        print("RNA: already present")
        return
    print("RNA (LIBSVM cod-rna, 488,565 x 8 after joining the three files)")
    parts = []
    for url in RNA_URLS:
        dest = os.path.join(HERE, os.path.basename(url))
        fetch(url, dest)
        parts.append(dest)
    with open(out, "wb") as f:
        for part in parts:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, f)


def main():
    wanted = sys.argv[1:] or ["song", "rna"]
    if "song" in wanted:
        get_song()
    if "rna" in wanted:
        get_rna()
    if not os.path.exists(os.path.join(HERE, "bio_train.dat")):
        print("KDD: download by hand (registration needed), see the note at the top of this file")


if __name__ == "__main__":
    main()
