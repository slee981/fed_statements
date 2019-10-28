#######################################################################
# Goal   : take raw text from fomc speeches and convert to dataframe

# Author : Stephen Lee 
# Date   : 9.23.19
#######################################################################

import pandas as pd 
from datetime import datetime, timedelta
import os, sys 

#######################################################################
# Storage 
#######################################################################

START_DATE = '2006-01-01'

ROOT_DIR = os.getcwd()
DATA_DIR = 'data'

RAW_ASSET_FILE = os.path.join(DATA_DIR, 'asset_prices.csv')
FOMC_DECISION_FILE = os.path.join(DATA_DIR, 'fomc_decisions.csv')
ASSET_DF_FILE = os.path.join(ROOT_DIR, DATA_DIR, 'assets.csv')

#######################################################################
# Main 
#######################################################################

if __name__ =='__main__': 

    # read in data 
        
    fomc_df = pd.read_csv(FOMC_DECISION_FILE)
    fomc_df['Date'] = pd.to_datetime(fomc_df['Release Date'])
    fomc_df = fomc_df.drop(['Release Date', 'Time'], axis=1).dropna()
    fomc_df = fomc_df[fomc_df['Date'] > START_DATE].sort_values(by='Date', ascending=False).reset_index()

    assets_df = pd.read_csv(RAW_ASSET_FILE)
    assets_df['Date'] = pd.to_datetime(assets_df['Row'])
    assets_df = assets_df.drop('Row', axis=1).dropna().reset_index()

    # calculate several changes: 
    # 1) change since last FOMC 
    # 2) event change around decision 
    #    -> 1 day 
    #    -> 3 day 
    #
    # Final dataframe should have columns 
    # Date    t_1yr_lastFOMC    t_1yr_1day_event    ...     {{asset}}_1day_event 
    # ----    --------------    ----------------            -------------------
    #   .            .                  .           ...                 . 
    #   .            .                  .           ...                 . 
    #   .            .                  .           ...                 . 
    
    t_1yr_changes_lastFomc = []
    t_5yr_changes_lastFomc = []
    t_10yr_changes_lastFomc = []
    t_30yr_changes_lastFomc = []
    implied_changes_lastFomc = []
    sp500_changes_lastFomc = []
    sp500f_changes_lastFomc = []
    fomc_dates = fomc_df['Date'].iloc[:-1].copy()

    for i, date in enumerate(fomc_dates): 

        last_fomc = fomc_df['Date'].iloc[i+1]
        print('Date {} : LastFOMC {}'.format(date, last_fomc))
        assert(date > last_fomc)

        # day_before = date - timedelta(days = 1)
        # day_after = date + timedelta(days = 1)
        # assert(day_after > date)
        # assert(day_before < date)

        t_1yr_change = assets_df['FCM1'].loc[assets_df['Date'] == date].item() - assets_df['FCM1'].loc[assets_df['Date'] == last_fomc].item()
        t_5yr_change = assets_df['FCM5'].loc[assets_df['Date'] == date].item() - assets_df['FCM5'].loc[assets_df['Date'] == last_fomc].item()
        t_10yr_change = assets_df['FCM10'].loc[assets_df['Date'] == date].item() - assets_df['FCM10'].loc[assets_df['Date'] == last_fomc].item()
        t_30yr_change = assets_df['FCM30'].loc[assets_df['Date'] == date].item() - assets_df['FCM30'].loc[assets_df['Date'] == last_fomc].item()
        implied_change = assets_df['PFNP'].loc[assets_df['Date'] == date].item() - assets_df['PFNP'].loc[assets_df['Date'] == last_fomc].item()
        sp500_change = assets_df['SP500'].loc[assets_df['Date'] == date].item() - assets_df['SP500'].loc[assets_df['Date'] == last_fomc].item()
        sp500f_change = assets_df['SPSPF'].loc[assets_df['Date'] == date].item() - assets_df['SPSPF'].loc[assets_df['Date'] == last_fomc].item()

        t_1yr_changes_lastFomc.append(t_1yr_change)
        t_5yr_changes_lastFomc.append(t_5yr_change)
        t_10yr_changes_lastFomc.append(t_10yr_change)
        t_30yr_changes_lastFomc.append(t_30yr_change)
        implied_changes_lastFomc.append(implied_change)
        sp500_changes_lastFomc.append(sp500_change)
        sp500f_changes_lastFomc.append(sp500f_change)

    data = {
        'Date': fomc_dates, 
        'Change 1yr Treasury': t_1yr_changes_lastFomc, 
        'Change 5yr Treasury': t_5yr_changes_lastFomc, 
        'Change 10yr Treasury': t_10yr_changes_lastFomc, 
        'Change 30yr Treasury': t_30yr_changes_lastFomc, 
        'Change Implied Fed Fund': implied_changes_lastFomc, 
        'Change SP500': sp500_changes_lastFomc, 
        'Change SP500 Finance': sp500f_changes_lastFomc
    }

    df = pd.DataFrame(data)
    df.to_csv(ASSET_DF_FILE)
