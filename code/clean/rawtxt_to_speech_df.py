#######################################################################
# Goal   : take raw text from fomc speeches and convert to dataframe

# Author : Stephen Lee
# Date   : 9.23.19

# Note   : run from repo's root dir ./
#######################################################################

import pandas as pd
from datetime import datetime
import os, sys

#######################################################################
# Storage
#######################################################################

# run from repo root dir ./
ROOT_DIR = os.getcwd()
DATA_DIR = "data"

RAW_DATA_FOLDER = os.path.join(DATA_DIR, "fed_board")
DF_FILE = os.path.join(ROOT_DIR, DATA_DIR, "speeches_df.csv")

TXT_FILES = os.listdir(RAW_DATA_FOLDER)

#######################################################################
# Main
#######################################################################

if __name__ == "__main__":

    # bins to populate
    dates = []
    speakers = []
    text = []

    # Loop through the data folder and read in all the articles
    for fname in TXT_FILES:
        if ".txt" in fname:
            print(fname)
            fpath = os.path.join(RAW_DATA_FOLDER, fname)
        else:
            continue

        with open(fpath, "r", encoding="utf8") as f:
            txt = f.read()

        info = txt.split("|")

        # info is separated as date | speaker | text
        # this appraoch handles the text that also contains | for math stuff
        date = info[0]
        speaker = info[1]
        raw_txt = " ".join(info[2:])
        date = datetime.strptime(date, "%B %d, %Y")

        clean_txt = raw_txt.split("\n")[15:]  # <- remove header info
        clean_txt = " ".join(clean_txt)

        dates.append(date)
        speakers.append(speaker)
        text.append(clean_txt)

    # save data
    data = {"Date": dates, "Speaker": speaker, "Text": text}

    df = pd.DataFrame(data)
    df.to_csv(DF_FILE)
