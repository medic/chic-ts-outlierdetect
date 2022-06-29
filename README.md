# Time Series Forecasting for Outlier Detection

This tool aims to offer efficient exploration and selection of time series models for performance monitoring alert systems. It provides an interface and test bench where data scientists and developers can quickly implement and analyze univariate time series forecasting models. There is also a script that generates a configurable comparison of candidate models' performance.

An example use case for this work is monitoring CHW activity over time. A general, but useful, impact measure in community health systems would be number of forms or number of caring activities recorded for a region sampled on a weekly/monthly basis. In this example, we use that region's historical values and trends to predict future values. When the predictions significantly deviate from the actual values, we can send alerts to enable earlier intervention when the community health system is operating outside of the expected range.

## Methodology

This work assumes that a time series historical signal can be reasonably approximated using time series forecasting models. Our hypothesis is that if you can obtain a good fit between the signal and the forecasting model, the residual between the estimated signal and the true signal can help identify signal anomalies.  We particularly focus on more explainable and straightforward models including the [naive model](https://otexts.com/fpp2/simple-methods.html#na%C3%AFve-method), [moving mean](https://en.wikipedia.org/wiki/Moving_average) and [simple exponential smoothing](https://otexts.com/fpp2/simple-methods.html#na%C3%AFve-method). The module applies these models using a sliding window (the signal processing equivalent of applying low-pass FIR filters).  Outliers are identified by comparing the residual's magnitude to it's historical statistical distribution and by default defined as occurring when the residual falls outside of the sliding window's prediction 95% confidence interval.

![](img/time_series_model.jpg)

## Installation/Set-up

To use this tool, set up and activate a Python virtual environment. Instructions to install, create and activate a virtual environment can be found here: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/. Once the virtual environment is activated, install the dependencies using the following command:

	python3 install -r requirements.txt

This work was developed using python 3.8.

## Run instructions

### Input data

An example of the suggested input file is shown in example.csv. This example data models a performance cost at a weekly basis. The first column should be able to be cast by the [pandas.to_datetime()](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html) method. This work has been tested with weekly/monthly data. The second column should be the data of interest.

### Example Single Model Run

Once set-up is complete, you can run an example analysis:

	python3 example.py

This script shows how to run an experiment for the naive time series model. The output of the test run are written to the out/example directory.

### Model Comparison

There is a script to aid with multiple time series model comparisons:

	python3 compare_models.py

The comparison script fits all time series models defined by the config.yaml file, stores the individual test runs in the out/ directory and also writes a csv file showing summary metrics of the model's residual (error) analysis to out/res_stats_all.csv. The example configuration sets up an experiment that compares the naive model, equally weighted moving average and simple exponential smoothing with a sliding window of 16 weeks.

The model comparison functionality is achieved by setting up the config.yaml file. The config file defines the input file, write specifications and the parameters for each model run. The example config file is shown below.

```
data_file: "example.csv"
write_output: True
experiments:
    - name: "example_naive"
      model_class: TimeSeriesNaiveModel
      train_periods: 16
      model_kwargs:
      fit_kwargs:
    - name: "example_exponentialsmoothing"
      model_class: TimeSeriesExpSmoothingModel
      train_periods: 16
      model_kwargs:
          initialization_method : estimated
      fit_kwargs:
          smoothing_level : 0.5
```

In order to set up your own experiment, you would make a copy of the example config file and change the parameters accordingly. Finally, you could update the compare_models.py script to read yoru new config file. The config parameters are described below:

- data_file: filepath of the time series data (specifications given in "Input data" section)
- write_output: specifies if the experiment run should write data to the ../out directory
- experiments: list of different time series models to test
  - name: name of directory to write experiment results to
  - model_class: type of model class (see ts_outlierdetect/ts_univariate_outlier.py for available options)
  - train_periods: integer size of sliding window
  - model_kwargs: dictionary of kwargs to pass to the model_class constructor
  - fit_kwargs: dictionary of kwargs to pass to the model_class's fit method

## Output

For each model run, the outputs are:

- residual_data.csv: Time series data including estimated signal, residual signal, percent error
- residual_stats.csv: Residual analysis statistics including MAE, MAPE and RMSE
- plots/autocorr.jpg: Residual autocorrelation plot
- plots/residual_histogram.jpg: Histogram of residual signal
- plots/residual.jpg: Residual time signal
- plots/time_series_model.jpg: Comparison of true and estimated signal with outliers identified

The output for a comparison run (as shown in compare_models.py) includes a directory for each model experiment as described above, as well as:

- res_stats_all.csv: Residual analysis statistics for all model runs

## Final Model Selection

After the model comparison experiment is complete, final selection of the best alert/outlier detection system is largely dependent on the use case and implementer preference. For early implementations, we recommend optimizing for the minimum Minimum absolute error (MAE) or minimum absolute percentage error (MAPE), which indicates better model fit and penalizes less for outliers. These performance metrics are located in the out/res_stats_all.csv file.

For further improvements, we recommend measuring alert system performance similar to a supervised binary classifier. This would include labeling known datasets with outliers you wish to be notified about, running the different model configurations and comparing the system's indicated outliers (located in residual_data.csv) to the labeled dataset to measure the alert system's precision and recall.

## Time Series Model Customization

If you wish to build a custom model that takes advantage of some of the functionality of this module, you can implement that by building a class that inherits from the base TimeSeriesUnivariateModel class as shown in  ts_outlierdetect/ts_univariate_outlier.py and implementing the constructor, fit and predict_next methods. The documentation inside the code will be the best reference for how to achieve this, as well as referencing the simple TimeSeriesNaiveModel model.
