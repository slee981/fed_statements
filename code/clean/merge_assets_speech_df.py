#######################################################################
# Goal   : take dataframe with date | speaker | text and add a column
#          with the 1, 2, and 3 gram versions

# Author : Stephen Lee
# Date   : 9.23.19
#######################################################################

import pandas as pd
from datetime import datetime
import os, sys

###########################################################################
# Storage
###########################################################################

ROOT_DIR = os.getcwd()
DATA_DIR = "data"
SPEECHES = os.path.join(ROOT_DIR, DATA_DIR, "speeches_ngrams_df.csv")
FOMC_DECISIONS = os.path.join(ROOT_DIR, DATA_DIR, "fomc_decisions_df.csv")
ASSETS = os.path.join(ROOT_DIR, DATA_DIR, "assets_df.csv")
STOP_WORDS = os.path.join(ROOT_DIR, "code", "clean", "stopwords.txt")
MASTER_DF_FILE = os.path.join(ROOT_DIR, DATA_DIR, "ngrams_assets_df.csv")
FIRST_DATE_IN_SET = "2006-01-01"

###########################################################################
# Functions
###########################################################################


def get_speeches_between_meetings(speeches_df, fomc_df):
    """
    INPUT    -> dfs of fomc speeches and fomc decisions
    OUTPUT   -> df of fomc decision dates and text from every 
                speech given since the last meeting 
    """
    # WARNING: this is going to get ugly

    # we have dataframes with different dates, speeches and fomc meetings.
    # we want to keep all text (2-grams) that occurs between meeting dates.

    raw_speech_between_meetings = []
    speeches_between_meetings = []
    speech_dates_between_meetings = []
    meeting_dates_used = []

    raw_text = speeches_df["stemmed"].copy()
    speech_text = speeches_df["2-gram"].copy()
    speech_dates = speeches_df["Date"].copy()
    meeting_dates = fomc_df["Date"].copy()

    # loop through the fomc dates
    # note that this is tricky because
    # we want to grab both the current date and the next date to compare
    num_speeches = len(speech_text)
    speech_idx = 0
    for idx in range(0, len(meeting_dates) - 1):
        current_meeting_date = meeting_dates.iloc[idx]
        previous_meeting_date = meeting_dates.iloc[idx + 1]
        assert current_meeting_date > previous_meeting_date

        all_raw_speech_this_round = " "
        all_speech_this_round = " "
        all_speech_dates_this_round = []

        # since we sorted speeches by date, we can keep track of the index
        # so we can do an O(n+m) instead of O(nm)

        # now we want to get all text that happened
        # -> BEFORE the current date, and
        # -> AFTER the previous dates
        print("--------------------------------------------")
        print("Meeting date {}".format(current_meeting_date))
        print("Prev date {}".format(previous_meeting_date))
    
        while (
            (speech_idx < num_speeches)
            and (speech_dates.iloc[speech_idx] < current_meeting_date)
            and (speech_dates.iloc[speech_idx] >= previous_meeting_date)
        ):
            print("    Speech date {}".format(speech_dates.iloc[speech_idx]))
            speech = speech_text.iloc[speech_idx] + "."
            date = speech_dates.iloc[speech_idx]
            raw_speech = raw_text.iloc[speech_idx] + " "

            all_raw_speech_this_round += raw_speech
            all_speech_this_round += speech
            all_speech_dates_this_round.append(date)
            speech_idx += 1

        raw_speech_between_meetings.append(all_raw_speech_this_round)
        speeches_between_meetings.append(all_speech_this_round)
        speech_dates_between_meetings.append(all_speech_dates_this_round)
        meeting_dates_used.append(current_meeting_date)

    print("\n\nFinal info -----------------------------")
    print("Length meeting dates used: {}".format(len(meeting_dates_used)))
    print("Length speeches: {}".format(len(speeches_between_meetings)))
    print("Length dates: {}".format(len(speech_dates_between_meetings)))

    # tie all series together
    data = {
        "Date": meeting_dates_used,
        "2-gram": speeches_between_meetings,
        "Speech Dates": speech_dates_between_meetings,
        "Stemmed": raw_speech_between_meetings,
    }
    return pd.DataFrame(data)


###########################################################################
# Main
###########################################################################

if __name__ == "__main__":

    # prepare dataframes
    speeches_df = pd.read_csv(
        SPEECHES, usecols=["Date", "stemmed", "2-gram"]
    ).drop_duplicates(subset="2-gram")
    speeches_df["Date"] = pd.to_datetime(speeches_df["Date"])
    speeches_df = speeches_df.sort_values(by="Date", ascending=False).reset_index()

    fomc_df = pd.read_csv(FOMC_DECISIONS)
    fomc_df["Date"] = pd.to_datetime(fomc_df["Release Date"])
    fomc_df = fomc_df.drop(["Release Date", "Time"], axis=1).dropna()
    fomc_df = (
        fomc_df[fomc_df["Date"] > FIRST_DATE_IN_SET]
        .sort_values(by="Date", ascending=False)
        .reset_index()
    )

    assets_df = pd.read_csv(ASSETS).drop("Unnamed: 0", axis=1)
    assets_df["Date"] = pd.to_datetime(assets_df["Date"])

    # process

    # get the speech info
    speeches_between_meetings_df = get_speeches_between_meetings(speeches_df, fomc_df)

    # merge data
    df = fomc_df.copy()
    df = df.merge(speeches_between_meetings_df, on="Date").merge(assets_df, on="Date")

    print(df.columns)
    df.to_csv(MASTER_DF_FILE)
