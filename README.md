# Time Series Forecasting Tool Analysis and Comparison

This tool provides an interface and test bench for various time series forecasting models. To date, this module focuses on univariate time series data. The work's main objective is to provide an abstract interface where data scientists and developers can quickly implement a time series model and automatically generate some performance metrics and plots. There is also a script that generates a performance comparison for pre-defined configurable candidate models.

The motivation for this work is to enable efficient exploration and selection of time series models for performance monitoring alert systems. An example of a general, but useful, impact measure in community health systems would be number of forms or number of caring activities recorded for a region sampled on a weekly/monthly basis. In this example, we wish to implement a model that uses historical values to predict future values, alerting users and enabling earlier intervention when the community health system is operating outside of the expected range. The alert system performance is dependent on the time series model's fit accuracy. There are a plethora of time series forecasting options and optimal selection is highly dependent on the data itself. This module offer some of the straightforward models (e.g. naive, rolling window average, exponential smoothing) for quick testing. Furthermore, it is easily adapted for most time series forecasting methods in the [statsmodels](https://www.statsmodels.org/dev/tsa.html) library.

The methodology of this library is based on the hypothesis that if you can obtain a good fit of a time series analysis/forecasting model, the residual between the estimated signal and the true signal can help identify signal anomalies. Currently, this tool can assign outliers by comparing the residual's magnitude to it's historical statistical distribution or the original signal's magnitude.

![ ](img/time_series_model.jpg)

## Installation/Set-up

To use this tool, set up and activate a Python virtual environment. Instructions to install, create and activate a virtual environment can be found here: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/. Once the virtual environment is activated, install the dependencies using the following command:

	python3 install -r requirements.txt


## Run instructions

### Input data

An example of the suggested input file is shown in example.csv. This example data models a performance cost at a weekly basis. The first column should be able to be cast by the pandas.to_datetime()[https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html] method. This work has been tested with weekly/monthly data. The second column should be the data of interest.

### Example Single Model Run

Once set-up is complete, you can run an example analysis:

	python3 example.py

This script shows how to run an experiment for the naive time series model. The output of the test run are written to the out/example_naive directory.

### Model Comparison

There is also a script to aid with multiple time series model comparisons:

	python3 compare_models.py

The comparison script fits all time series models defined by the config.yaml file, stores the individual test runs in the out/ directory and also writes a csv file showing summary metrics of the model's residual (error) analysis to out/res_stats_all.csv.

The model comparison functionality is achieved by setting up the conig.yaml file. The config file defines the input file, write specifications and the parameters for each model run.

## Output

The output for a single test run (as shown in example.py) are:

- residual_data.csv: Time series data including estimated signal, residual signal, percent error
- residual_stats.csv: Residual analysis statistics including MAE, MAPE and RMSE
- plots/autocorr.jpg: Residual autocorrelation plot
- plots/residual_histogram.jpg: Histogram of residual signal
- plots/residual.jpg: Residual time signal
- plots/time_series_model.jpg: Comparison of true and estimated signal with outliers identified

The output for a comparison run (as shown in compare_models.py) includes a directory for each model experiment as described above, as well as:

- res_stats_all.csv: Residual analysis statistics for all model runs

Note that when comparing time series models for alert/outlier detection systems, we recommend using MAE or MAPE which penalize less for the anomalies.

## Time Series Model Customization

If you wish to build a custom model that takes advantage of some of the fucntionality of this module, you can implement that be building a class that inherits from the base TimeSeriesUnivariateModel class as shown in  ts_outlierdetect/ts_univariate_outlier.py and implementing the constructor and fit method. The documentation inside the code will be the best reference for how to achieve this, as well as referencing the simple TimeSeriesNaiveModel model.
