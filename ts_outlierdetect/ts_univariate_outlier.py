from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import statsmodels.tsa.api as tsa


logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class TimeSeriesUnivariateModel(object):
    """
    Time series univariate model interface/base class
    """
    
    def __init__(self, window_alert_n=None, output_dir=None, write_output=True):
        """
        Class constructor
        
        Parameters
        ----------
        window_alert_n: int, default None
            Number of data points to use for training stastical profiling alert model.
            If None, uses entire signal.
        output_dir: str
            Directory for output data and plots in 'out' folder
        write_output: bool, default True
            If True, plots and data are written to file.
        
        Returns
        -------
        None
        """
        #Initialize a data dataframe containing model's data
        self.data = pd.DataFrame(columns=['y', 'y_hat', 'res'])
        #Initialize a res_stats dict for model error stats
        self.res_stats = {}
        #Define rolling window size for anamoly/alert detection
        self._window_alert_n = window_alert_n
        #Define output dir and writing capabilities
        self._write_output = write_output
        if self._write_output:
            if output_dir == None:
                self._output_dir = Path.cwd() / 'out'
            else:
                self._output_dir = Path.cwd() / 'out' / output_dir
            if not self._output_dir.exists():
                Path.mkdir(self._output_dir, parents=True)
                Path.mkdir(self._output_dir / 'plots')

            
    def residual_analysis(self, outlier_kwargs=None):
        """
        Calculates model residual (error) statistics and saves them to self.res_stats.
        
        Parameters
        ----------
        outlier_kwargs: dict, default None
            keyword arguments to pass to self.find_historical_outliers_idx method
        
        Returns
        -------
        None
        """
        logging.info('Performing residual analysis and outlier detection.')
        #Calculate residual signal
        self.data['res'] = self.data['y'] - self.data['y_hat']
        #Calculate the MAPE of the residual
        self.data['percent_error'] = self.data['res']/self.data['y']
        #Calcuate basic stats about residual
        self.res_stats['mu'] = np.mean(self.data['res'])
        self.res_stats['std'] = np.std(self.data['res'])
        self.res_stats['MAE'] = np.mean(np.abs(self.data['res']))
        self.res_stats['MAPE'] = np.mean(np.abs(self.data['res'])/self.data['y'])
        self.res_stats['RMSE'] = np.sqrt(np.mean(np.square(self.data['res'])))
        if outlier_kwargs != None:
            self.find_historical_outliers_idx(**outlier_kwargs)
        else:
            self.find_historical_outliers_idx()
        self.data['outlier'] = self.data.index.isin(self._outliers_index)
        #Save data to file
        if self._write_output:
            output_resanalysis_filepath = self._output_dir / 'residual_data.csv'
            self.data.to_csv(output_resanalysis_filepath)
            output_resstats_filepath = self._output_dir / 'residual_stats.csv'
            pd.DataFrame.from_dict(self.res_stats, orient='index').to_csv(output_resstats_filepath, header=False)
            logging.info('Residual analysis and outlier detection results written to {} and {}'.format(output_resanalysis_filepath,output_resstats_filepath))
        
        
    def plot_model_results(self):
        """
        Plots different model visualizations including fitted time series signal,
        model residual and autocorrelation plot.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        logging.info('Plotting time series anomoly analysis')
        self._plot_time_series_model()
        self._plot_res()
        self._plot_res_hist()
        self._plot_autocorrelation()
        if self._write_output:
            plots_dir = self._output_dir / 'plots'
            logging.info('Saving plots to: {}'.format(plots_dir))
        
        
        
    def _plot_time_series_model(self, filename=None, plot_outliers=True, plot_ci=True):
        """
        Plots raw time series and model estimates with 95% confidence intervals.
        
        Parameters
        ----------
        filename: str or Path, default None 
            Filename for output plot. If None, plot is saved as ../out/plots/time_series_model.png.
        plot_outliers: bool, default True
            If True, outliers as defined by model are shown on the plot.
        plot_ci: bool, default True
            If True, the 95% confidence interval is shown on the plot.
            
        Returns
        -------
        None
        """
        fig, ax = plt.subplots()
        ax.plot(self.data.index, self.data['y'], color="black", label = 'y')
        ax.plot(self.data.index, self.data['y_hat'], color="blue", label = 'y_hat')
        ax.figure.autofmt_xdate()
        ax.legend()
        ax.fill_between(self.data['y_hat'].index, (self.data['y_hat'].values+self.data['res_outlier_ub'].values), (self.data['y_hat'].values+self.data['res_outlier_lb'].values), color='b', alpha=.1)
        if plot_outliers and self._outliers_index is not None:
                ax.scatter(self._outliers_index.values, self.data.loc[self._outliers_index]['y'].values, color="red", label = 'outliers')
        #Save plot to file
        if self._write_output:
            if filename == None:
                filename = self._output_dir / 'plots' /  'time_series_model.jpg'
            plt.savefig(filename)
        return None
        
    def _plot_res(self, filename=None, plot_outliers=True, plot_ci=True):
        """
        Plots model residual (error).
        
        Parameters
        ----------
        filename: str or Path, default None 
            Filename for output plot. If None, plot is saved as ../out/plots/residual.png.
        plot_outliers: bool, default True
            If True, outliers as defined by model are shown on the plot.
        plot_ci: bool, default True
            If True, the 95% confidence interval is shown on the plot.
            
        Returns
        -------
        None
        """
        fig, ax = plt.subplots()
        ax.plot(self.data.index, self.data['res'], color="black", label = 'residual')
        ax.figure.autofmt_xdate()
        ax.legend()
        ax.fill_between(self.data['res'].index, self.data['res_outlier_lb'], self.data['res_outlier_ub'], color='b', alpha=.1)
        if plot_outliers and self._outliers_index is not None:
            ax.scatter(self._outliers_index.values, self.data.loc[self._outliers_index]['res'].values, color="red", label = 'outliers')
        #Save plot to file
        if self._write_output:
            if filename == None:
                filename = self._output_dir / 'plots' /  'residual.jpg'
            plt.savefig(filename)
        return None
        
    def _plot_res_hist(self, filename=None):
        """
        Plots model residual (error) histogram.
        
        Parameters
        ----------
        filename: str or Path, default None 
            Filename for output plot. If None, plot is saved as ../out/plots/residual.png.
            
        Returns
        -------
        None
        """
        fig, ax = plt.subplots()
        ax.hist(self.data['res'], color="blue", label = 'residual')
        ax.legend()
        #Save plot to file
        if self._write_output:
            if filename == None:
                filename = self._output_dir / 'plots' /  'residual_histogram.jpg'
            plt.savefig(filename)
        return None
        
    def _plot_autocorrelation(self, col='res', filename=None):
        """
        Plots autocorrelation plot.
        
        Parameters
        ----------
        col: str, default 'res'
            Column name for variable to plot autocorrelation
        filename: str or Path, default None 
            Filename for output plot. If None, plot is saved as ../out/plots/autocorr.png.
            
        Returns
        -------
        None
        """
        fig, ax = plt.subplots()
        pd.plotting.autocorrelation_plot(self.data[col].dropna(), ax=ax)
        ax.set_title('Autocorrelation Plot')
        if self._write_output:
            if filename == None:
                filename = self._output_dir / 'plots' /  'autocorr.jpg'
            plt.savefig(filename)
        return None
                
    def find_historical_outliers_idx(self, th_type='std_res', percent_th=.25, std_th=1.96):
        """
        Finds outliers in historical data. Outliers are those whose model estimate
        falls outside a predetermined threshold. The threshold type can be defined as 
        95% confidence interval, a constant multiple of the residual's standard deviation
        or a percentage of the raw signal's value.
        
        Parameters
        ----------
        th_type: str, default "std_res"
            Defined the thresholding mechanism. If 'std_res', an outlier is defined as a constant multiple
            of the residual's standard deviation. If 'percent_raw', an outlier is defined as
            a percent of the raw time signal's value.
        percent_th: float, default 0.25
            If th_type=='percent_raw', a point in the raw time series, y(t), is classified
            as an outlier if the absolute value of the residual at that point, res(t), is
            greater than percent_th*y(t).
        std_th: float, default 1.96
            If th_type='std_res', a point in the raw time series, y(t), is classified as
            as outlier if the value of the residual, res(t), is greater than std_th*np.std(res).
            
        Returns
        -------
        None
        """        
        if th_type=='std_res':
            if self._window_alert_n is not None:
                self.data['res_outlier_ub'] = self.data['res'].dropna().rolling(self._window_alert_n+1)\
                .apply(lambda x: x[:-1].mean()+std_th*x[:-1].std())
                self.data['res_outlier_lb'] = self.data['res'].dropna().rolling(self._window_alert_n+1)\
                .apply(lambda x: x[:-1].mean()-std_th*x[:-1].std())
            else:
                self.data['res_outlier_ub'] = self.res_stats['mu'] + std_th*self.res_stats['std']
                self.data['res_outlier_lb'] = self.res_stats['mu'] - std_th*self.res_stats['std']
        elif th_type=='percent_raw':
            self.data['res_outlier_ub'] = self.data['y'] + self.data['y']*percent_th
            self.data['res_outlier_lb'] = -1.0*self.data['y']*percent_th
            
        self._outliers_index = self.data.loc[(self.data['res'] > self.data['res_outlier_ub']) 
        | (self.data['res'] < self.data['res_outlier_lb'])].index
        
        return None
        
        
    @abstractmethod
    def fit(self, ts:pd.Series):
        """
        Method for fitting the model to the time series data. When implementing this method, it
        is expected that the method will at a minimum perform the following actions:
        1. Assign the time series data to self.data['y'].
        2. Assign model estimates to self.data['y_hat'].
        3. Runs the TimeSeriesUnivariateModel.residual_analysis() method.
        
        See Also
        --------
        TimeSeriesNaiveModel.fit() : Shows a simple implementation of a naive model's fit function.
        """
        pass
        
    
class TimeSeriesNaiveModel(TimeSeriesUnivariateModel):
    """
    Naive time series univariate model. For this model, the estimated value at each time step
    is defined as the previous value: y_hat(t) = y(t-1).
    """
    
    def __init__(self, window_alert_n=None, output_dir=None, write_output=True):
        """
        Class constructor
        
        Parameters
        ----------
        window_alert_n: int, default None
            Number of data points to use for training stastical profiling alert model.
            If None, uses entire signal.
        output_dir: str or Path
            Directory for output data and plots
        write_output: bool, default True
            If True, plots and data are written to file.
        
        Returns
        -------
        None
        """
        super().__init__(window_alert_n=window_alert_n, output_dir=output_dir, write_output=write_output)
        
    def fit(self, ts:pd.Series):
        """
        Fits naive time series univariate model, y_hat(t) = y(t-1).
        
        Parameters
        ----------
        ts: pd.Series
            Raw time series data with a DatimeTimeIndex
        
        Returns
        -------
        None
        """
        logging.info('Fitting the naive time series model')
        #Assign raw time series value
        self.data['y'] = ts
        #Calc yhat
        self.data['y_hat'] = self.data['y'].shift(1)
        #Calculate basic residual signal stats
        self.residual_analysis()
        
        
class StatsModelTSAModel(TimeSeriesUnivariateModel):
    """
    Time series univariate model to implement models from the statsmodels tsa library.
    
    
    See Also
    --------
    https://www.statsmodels.org/dev/tsa.html#module-statsmodels.tsa
    """
    
    def __init__(self, window_alert_n=None, output_dir=None, write_output=True, tsa_model=tsa.SimpleExpSmoothing, tsa_kwargs={'initialization_method':"estimated"}):
        """
        Class constructor
        
        Parameters
        ----------
        window_alert_n: int, default None
            Number of data points to use for training stastical profiling alert model.
            If None, uses entire signal.
        output_dir: str or Path
            Directory for output data and plots
        write_output: bool, default True
            If True, plots and data are written to file.
        tsa_model: obj or str, default tsa.SimpleExpSmoothing
            Type of model as defined by the statsmodels.tsa API
        tsa_kwargs: dict
            Keyword arguments passed to statsmodels.tsa.tsa_model
            at instantiation
        
        Returns
        -------
        None
        """
        super().__init__(window_alert_n=window_alert_n, output_dir=output_dir, write_output=write_output)
        if isinstance(tsa_model, str):
           self._tsa_model = getattr(tsa, tsa_model)
        else:
            self._tsa_model=tsa_model
        self._tsa_kwargs = tsa_kwargs
        
    def fit(self, ts:pd.Series, tsa_fit_kwargs=None):
        """
        Fits model from statsmodels time series analysis library.
        
        Parameters
        ----------
        ts: pd.Series
            Raw time series data with a DatimeTimeIndex
        tsa_fit_kwargs: dict
            Keyword arguments passed to statsmodels.tsa.tsa_model.fit()
            method
        
        Returns
        -------
        None
        """
        logging.info('Fitting a statmodels tsa library model')
        #Assign raw time series value
        self.data['y'] = ts
        #Assign the tsa fit kwargs
        self._tsa_fit_kwargs = tsa_fit_kwargs
        
        #Calc yhat
        if self._tsa_fit_kwargs != None:
            self.data['y_hat'] = self._tsa_model(self.data['y'], **self._tsa_kwargs).fit(**self._tsa_fit_kwargs).fittedvalues
        else:
            self.data['y_hat'] = self._tsa_model(self.data['y'], **self._tsa_kwargs).fit().fittedvalues
        #Calculate basic residual signal stats
        self.residual_analysis()
        
        
class TimeSeriesUnivariateRollingWindow(TimeSeriesUnivariateModel):
    """
    Time series univariate model using a equally weighted rolling window mean
    implemented using the pd.DataFrame.rolling method and is configurable for
    all kwargs of that method (e.g. window length, cenetered vs. historical-only).
    
    See Also:
    ---------
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    
    """
    
    def __init__(self, window_alert_n=None, output_dir=None, write_output=True):
        """
        Class constructor
        
        Parameters
        ----------
        window_alert_n: int, default None
            Number of data points to use for training stastical profiling alert model.
            If None, uses entire signal.
        output_dir: str or Path
            Directory for output data and plots
        write_output: bool, default True
            If True, plots and data are written to file.
        
        Returns
        -------
        None
        """
        super().__init__(window_alert_n=window_alert_n,output_dir=output_dir, write_output=write_output)
        
    def fit(self, ts:pd.Series, pandas_rolling_kwargs={'window':3,'center': False}):
        """
        Fits model from statsmodels time series analysis library.
        
        Parameters
        ----------
        ts: pd.Series
            Raw time series data with a DatimeTimeIndex
        pandas_rolling_kwargs: dict
            Keyword arguments to pass to the pd.DataFrame.rolling method
        
        Returns
        -------
        None
        """
        logging.info('Fitting a rolling average mean window')
        #Assign raw time series value
        self.data['y'] = ts
        #Calc yhat
        self.data['y_hat'] = self.data['y'].rolling(**pandas_rolling_kwargs).mean().shift(1)
        #Calculate basic residual signal stats
        self.residual_analysis()
        