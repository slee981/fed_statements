library(tidyverse)
library(broom)

rm(list=ls())

######################################################################
# FUNCTIONS 

results_to_latex <- function(fname, ...){
    tex_output <- capture.output(stargazer(...))
    cat(paste(tex_output, collapse='\n'), '\n', file=fname)
}

######################################################################
# MAIN

# get asset prices 
df_assets <- read_csv('data/assets.csv')
df_assets <- df_assets %>% select(-c('X1'))

# get fomc decisions, convert date from string to Date object
df_fomc <- read_csv('data/fomc_decisions.csv')
df_fomc$Date <- as.Date(df_fomc$`Release Date`, format='%d-%b-%y')

# merge interest rate decision on date
df <- merge(df_fomc, df_assets, by='Date', all=FALSE)
df <- df %>% select(-c('Release Date', 'Previous', 'Time'))  # drop stupid columns 

# convert to bps
df$ChangeBPS <- df$Change * 10000                            # stored in decimal      
df$ChangeImpliedBPS <- df$`Change Implied Fed Fund` * 100    # stored in pct
df$Change1YrBPS <- df$`Change 1yr Treasury` * 100            # stored in pct

# run simple OLS 

# Change ~ ImpliedFedFunds
res_uni <- lm(df$ChangeBPS ~ df$ChangeImpliedBPS)

# Change ~ ImpliedFedFunds + 1Yr Treasury 
res_multi <- lm(df$ChangeBPS ~ df$ChangeImpliedBPS + df$Change1YrBPS)

# Export results 

# csv 

# res_table <- tidy(res)                    
# write.csv(res_table, 'singleOLS.csv')        

# latex 

results_to_latex('ols_univariate.txt', res_uni, title='Univariate OLS Regression')
results_to_latex('ols_multivariate.txt', res_multi, title='Multivariate OLS Regression')
