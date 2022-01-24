import pandas as pd
import logging

            
def load_time_series_from_csv(csv_filename, time_col=None):
    """
    Loads time series data from CSV file into a Pandas
    Series or DataFrame. The CSV file must contain a column
    that Pandas is able to (a) cast to a DateTime and (b) infer
    a frequency (e.g. weekly/monthly) from.
    
    Parameters
    ----------
    csv_filename: str or Path
        Filename of CSV file to load
    time_col: str, default None
        Column name containing the DateTime field.
        If none, assume it is the first column with no name.
    
    Returns
    -------
    None
    """
    logging.info('Loading time series data from: {}'.format(csv_filename))
    #Load data from CSV file
    if time_col == None:
        ts = pd.read_csv(csv_filename, header=None)
        time_col = 0
    else:
        ts = pd.read_csv(csv_filename)
    #Set index as the DateTime column specified
    ts[time_col] = pd.to_datetime(ts[time_col])
    ts.set_index(time_col, inplace=True)
    #define dataframe's time frequency, useful for later data viz/computation 
    index_values_new = ts.index.values
    index_freq = ts.index.inferred_freq
    ts.index = pd.DatetimeIndex(index_values_new, freq=index_freq)
    #return data
    return ts