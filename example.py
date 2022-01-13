from ts_outlierdetect import ts_univariate_outlier, util
import logging
import pandas as pd
from pathlib import Path
import statsmodels.tsa.api as tsa


if __name__ == '__main__':
    
    time_series_data = util.load_time_series_from_csv('example.csv', 'period_start')
    
    #Sample Experiment #1 - Naive Model
    #Define the parameters for teh test run
    output_dir = 'example_naive'
    #Instantiate the model class
    model = ts_univariate_outlier.TimeSeriesNaiveModel(output_dir)
    #Fit the model to the time series
    model.fit(time_series_data)
    #Plot the results
    model.plot_model_results()