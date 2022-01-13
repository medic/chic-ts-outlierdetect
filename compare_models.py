from ts_outlierdetect import ts_univariate_outlier, util
import yaml
from pathlib import Path
import logging
import pandas as pd


def compare_ts_univariate_outlier(config_file):
    """
    Compares different time series models for outlier detection as defined
    by a configuration file.
    
    Parameters
    ----------
    config_file: Path or str
        Filepath for config file that defines the different experiments to run
    
    Returns
    -------
    None
    
    """
    # Read config file
    with open(config_file, 'r') as stream:
        experiment_config = yaml.safe_load(stream)
    #load data file
    DATA_FILE = experiment_config['data_file']
    
    time_series_data = util.load_time_series_from_csv('example.csv', 'period_start')
    #Record write_file
    write_output = experiment_config['write_output']
    if write_output == None:
        write_output = True
        
    #Save residual stats from all runs
    res_stats_all = {}
    
    #Run all experiments
    for experiment in experiment_config['experiments']:
        logging.info('Running experiment: {}'.format(experiment['name']))
        output_dir = experiment['name']
        model_class = experiment['model_class']
        model_kwargs = experiment['model_kwargs']
        fit_kwargs = experiment['fit_kwargs']
        if model_kwargs != None:
            model = getattr(ts_univariate_outlier, model_class)(output_dir,write_output,**model_kwargs)
        else:
            model = getattr(ts_univariate_outlier, model_class)(output_dir,write_output)
        if fit_kwargs != None:
            model.fit(time_series_data, **fit_kwargs)
        else:
            model.fit(time_series_data)
        model.plot_model_results()
        #save stats from run
        res_stats_all[experiment['name']] = model.res_stats
        
    #Write the residual analysis statistics for all runs to file
    pd.DataFrame.from_dict(res_stats_all, orient='index').to_csv(Path.cwd() / 'out' / 'res_stats_all.csv')
      

if __name__ == '__main__':
    
    compare_ts_univariate_outlier("config.yaml")
        
        