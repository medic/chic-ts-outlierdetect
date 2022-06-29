from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import statsmodels.tsa.api as tsa
import sys


logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class TimeSeriesUnivariateModel(object):
    """
    Time series univariate model interface/base class
    """
    
    def __init__(self, train_periods=None, model_kwargs=None, fit_kwargs=None, output_dir=None, write_output=True):
        """
        Class constructor
        
        Parameters
        ----------
        train_periods: int, default None
            Number of data points to use for rolling window funcionality.
            If None, uses entire signal.
        model_kwargs: dict
            Keyword arguments to pass to model
        fit_kwargs: dict
            Keyword arguments to pass to model's fit method
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
        self._train_periods = train_periods
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
                
        #if no kwargs specified, create one
        self._model_kwargs = model_kwargs
        self._fit_kwargs = fit_kwargs

            
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
        self.data['relative_error'] = self.data['res']/self.data['y']
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
        
        
        
    def _plot_time_series_model(self, filename=None, plot_outliers=True):
        """
        Plots raw time series and model estimates with upper and lower bounds for outlier thresholds.
        
        Parameters
        ----------
        filename: str or Path, default None 
            Filename for output plot. If None, plot is saved as ../out/plots/time_series_model.png.
        plot_outliers: bool, default True
            If True, outliers as defined by model are shown on the plot.
            
        Returns
        -------
        None
        """
        fig, ax = plt.subplots()
        ax.plot(self.data.index, self.data['y'], color="black", label = 'y')
        ax.plot(self.data.index, self.data['y_hat'], color="blue", label = 'y_hat')
        ax.figure.autofmt_xdate()
        ax.legend()
        y_hat_plotting = self.data[['y_hat', 'res_outlier_ub', 'res_outlier_lb']].dropna().astype('float64')
        ax.fill_between(y_hat_plotting['y_hat'].index, (y_hat_plotting['y_hat'].values+y_hat_plotting['res_outlier_ub'].values), (y_hat_plotting['y_hat'].values+y_hat_plotting['res_outlier_lb'].values), color='b', alpha=.1)
        if plot_outliers and self._outliers_index is not None:
                ax.scatter(self._outliers_index.values, self.data.loc[self._outliers_index]['y'].values, color="red", label = 'outliers')
        #Save plot to file
        if self._write_output:
            if filename == None:
                filename = self._output_dir / 'plots' /  'time_series_model.jpg'
            plt.savefig(filename)
        return None
        
    def _plot_res(self, filename=None, plot_outliers=True):
        """
        Plots model residual (error).
        
        Parameters
        ----------
        filename: str or Path, default None 
            Filename for output plot. If None, plot is saved as ../out/plots/residual.png.
        plot_outliers: bool, default True
            If True, outliers as defined by model are shown on the plot.
            
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
            a percent of the raw time signal's value. If None, doesn't calculate upper and lower bounds
            and assumes it was done outside this function.
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
        if th_type is not None:
            if th_type=='std_res':
                self.data['res_outlier_ub'] = self.res_stats['mu'] + std_th*self.res_stats['std']
                self.data['res_outlier_lb'] = self.res_stats['mu'] - std_th*self.res_stats['std']
            elif th_type=='percent_raw':
                self.data['res_outlier_ub'] = self.data['y'] + self.data['y']*percent_th
                self.data['res_outlier_lb'] = -1.0*self.data['y']*percent_th
            
            
        self._outliers_index = self.data.loc[(self.data['res'] > self.data['res_outlier_ub']) 
        | (self.data['res'] < self.data['res_outlier_lb'])].index
        
        return None
        
    def _build_fit_window_model(self, ts:pd.Series):
        """
        Instantiates and fits same version of this model (with a window of
        the original time series signal)
        
        Parameters
        ----------
        ts: pd.Series
            Raw time series data with a DatimeTimeIndex
        
        Returns
        -------
        (TimeSeriesUnivariateModel) Same type model as the model that called this function,
        fitted to the provided time series signal
        """
        
        if self._model_kwargs is not None:
            if self._fit_kwargs is not None:
                model = getattr(sys.modules[__name__],self.__class__.__name__)(write_output=False,model_kwargs=self._model_kwargs, fit_kwargs=self._fit_kwargs)
            else:
                model = getattr(sys.modules[__name__],self.__class__.__name__)(write_output=False,model_kwargs=self._model_kwargs)
        else:
            if self._fit_kwargs is not None:
                model = getattr(sys.modules[__name__],self.__class__.__name__)(write_output=False, fit_kwargs=self._fit_kwargs)
            else:
                model = getattr(sys.modules[__name__],self.__class__.__name__)(write_output=False)
        model.fit(ts)
        return model
        
    def fit_rolling_window(self, ts:pd.Series):
        """
        Medthod to fitting and defining outliers using a rolling window.
        
        Parameters
        ----------
        ts: pd.Series
            Raw time series data with a DatimeTimeIndex
        
        Returns
        -------
        None
        """
        
        train_model = self._build_fit_window_model(ts.iloc[:self._train_periods])
        self.data['y'] = train_model.data['y']
        self.data['y_hat'] = [None]*len(self.data['y'])
        self.data['res_outlier_ub'] = [0]*len(self.data['y'])
        self.data['res_outlier_lb'] = [0]*len(self.data['y'])
        
        for k in range(self._train_periods,len(ts)):
            #predict the next step
            sub_model = self._build_fit_window_model((ts.iloc[k-self._train_periods:k]))
            y_k = ts.iloc[k].values[0]
            y_k_index = ts.index.values[k]
            y_hat_k = sub_model.predict_next()
            res = y_k-y_hat_k
            #Find the ub/lb confidence intervals for this window
            sub_model.residual_analysis()
            ub = sub_model.data['res_outlier_ub'].iloc[-1]
            lb = sub_model.data['res_outlier_lb'].iloc[-1]
            self.data = self.data.append(pd.DataFrame({'y':y_k,'y_hat':y_hat_k,'res': res, 'res_outlier_ub':ub, 'res_outlier_lb': lb}, index=pd.DatetimeIndex([y_k_index])))
        self.residual_analysis(outlier_kwargs={'th_type': None})
        
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
    
    def __init__(self, train_periods=None, output_dir=None, write_output=True):
        """
        Class constructor
        
        Parameters
        ----------
        train_periods: int, default None
            Number of data points to use for rolling training window.
            If None, uses entire signal.
        output_dir: str or Path
            Directory for output data and plots
        write_output: bool, default True
            If True, plots and data are written to file.
        
        Returns
        -------
        None
        """
        super().__init__(train_periods=train_periods, output_dir=output_dir, write_output=write_output)
        
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
        
    def predict_next(self):
        """
        Predict next point in time series signal, y_hat(t).
        Assumes the .fit() method has already been called.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        (float or int): Next forecasted value        
        """
        
        return self.data['y'].iloc[-1]

        
class TimeSeriesExpSmoothingModel(TimeSeriesUnivariateModel):
    """
    Time series exponential smoothing model.
    
    
    See Also
    --------
    https://www.statsmodels.org/dev/tsa.html#module-statsmodels.tsa
    """
    
    def __init__(self, train_periods=None, output_dir=None, write_output=True, model_kwargs=None, fit_kwargs=None):
        """
        Class constructor
        
        Parameters
        ----------
        train_periods: int, default None
            Number of data points to use for rolling training window.
            If None, uses entire signal.
        output_dir: str or Path
            Directory for output data and plots
        write_output: bool, default True
            If True, plots and data are written to file.
        model_kwargs: dict
            Keyword arguments passed to statsmodels.tsa.SimpleExpSmoothing
            at instantiation
        fit_kwargs: dict
            Keyword arguments passed to
            statsmodels.tsa.SimpleExpSmoothing.fit() method
        
        Returns
        -------
        None
        """
        super().__init__(train_periods=train_periods, model_kwargs=model_kwargs, fit_kwargs=fit_kwargs, output_dir=output_dir, write_output=write_output)
        #Assign the model kwargs
        self._model_kwargs = model_kwargs
        #Assign the tsa fit kwargs
        self._fit_kwargs= fit_kwargs
        
    def fit(self, ts:pd.Series):
        """
        Fits model from statsmodels time series analysis library.
        
        Parameters
        ----------
        ts: pd.Series
            Raw time series data with a DatimeTimeIndex
        
        Returns
        -------
        None
        """
        logging.info('Fitting a statmodels tsa library model')
        #Assign raw time series value
        self.data['y'] = ts

        #Assign the fitted model
        self._fitted_model = None
        
        #Calc yhat
        if self._fit_kwargs != None:
            if self._model_kwargs != None:
                self._fitted_model = tsa.SimpleExpSmoothing(self.data['y'], **self._model_kwargs).fit(**self._fit_kwargs)
            else:
                self._fitted_model = tsa.SimpleExpSmoothing(self.data['y']).fit(**self._fit_kwargs)
        else:
            if self._model_kwargs != None:
                self._fitted_model = tsa.SimpleExpSmoothing(self.data['y'], **self._model_kwargs).fit()
            else:
                self._fitted_model = tsa.SimpleExpSmoothing(self.data['y']).fit()
        #Find the fitted values, yhat
        self.data['y_hat'] = self._fitted_model.fittedvalues
        #Calculate basic residual signal stats
        self.residual_analysis()
        
    def predict_next(self):
        """
        Predict next point in time series signal.
        Assumes the .fit() method has already been called.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        (float or int): Next forecasted value        
        """
        return self._fitted_model.forecast(1)
        
class TimeSeriesMean(TimeSeriesUnivariateModel):
    """
    Time series univariate model using a equally weighted mean
    
    """
    
    def __init__(self, train_periods=None, output_dir=None, write_output=True):
        """
        Class constructor
        
        Parameters
        ----------
        train_periods: int, default None
            Number of data points to use for rolling training window.
            If None, uses entire signal.
        output_dir: str or Path
            Directory for output data and plots
        write_output: bool, default True
            If True, plots and data are written to file.
        
        Returns
        -------
        None
        """
        super().__init__(train_periods=train_periods,output_dir=output_dir, write_output=write_output)
        self._fit_kwargs = None
        self._model_kwargs = None
        
    def fit(self, ts:pd.Series):
        """
        Fits model from statsmodels time series analysis library.
        
        Parameters
        ----------
        ts: pd.Series
            Raw time series data with a DatimeTimeIndex
        
        Returns
        -------
        None
        """
        logging.info('Fitting a signal mean model')
        #Assign raw time series value
        self.data['y'] = ts
        #Calc yhat
        self.data['y_hat'] = self.data['y'].mean()
        #Calculate basic residual signal stats
        self.residual_analysis()
        
    def predict_next(self):
        """
        Predict next point in time series signal.
        Assumes the .fit() method has already been called.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        (float or int): Next forecasted value        
        """
        return self.data['y_hat'].iloc[-1]
        