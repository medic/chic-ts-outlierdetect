from ts_outlierdetect import ts_univariate_outlier, util
import logging
import os
import pandas as pd
from pathlib import Path
import statsmodels.tsa.api as tsa

'''
if __name__ == '__main__':
    
    time_series_data = util.load_time_series_from_csv('data/df_completed_tasks.csv')
    
    #Sample Experiment #1 - Naive Model
    #Define the parameters for the test run
    output_dir = 'example'
    #Instantiate the model class
    model = ts_univariate_outlier.TimeSeriesNaiveModel(output_dir=output_dir)
    #Fit the model to the time series
    model.fit(time_series_data)
    #Plot the results
    model.plot_model_results()
'''

if __name__ == '__main__':
    my_files= ["data/df__mal_pos.csv",
    "data/df_ari_cough.csv",
    "data/df_completed_task_rate.csv",
    "data/df_completed_tasks.csv",
    "data/df_mal_ds.csv",
    "data/df_mal_ds_rdt_yes.csv",
    "data/df_mal_ds_tested.csv",
    "data/df_rdt_tests.csv",
    "data/df_symptom_cough.csv",
    "data/df_symptom_fast_breath.csv",
    "data/df_total_tasks.csv",
    "data/df_treat_ari.csv",
    "data/df_u5_diarrhea.csv",
    "data/df_u5_diarrhea_treated.csv",
    "data/df_u5_rdt_done.csv",
    "data/df_u5_rdt_pos.csv"]
    for file in my_files:
        print(file)
        time_series_data = util.load_time_series_from_csv(file)
        #Sample Experiment #1 - Naive Model
        #Define the parameters for the test run
        output_dir = os.path.splitext(os.path.split(file)[1])[0]
        #Instantiate the model class
        model = ts_univariate_outlier.TimeSeriesNaiveModel(output_dir=output_dir)
        #Fit the model to the time series
        model.fit(time_series_data)
        #Plot the results
        model.plot_model_results()
