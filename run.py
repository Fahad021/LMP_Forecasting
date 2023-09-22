import pandas as pd
import numpy as np
import argparse
import os
import logging
import traceback
import sys

from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.models import LEAR

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default='PJM',
        help='Market under study. If not one of the standard ones, provide a CSV file name.'
    )
    parser.add_argument(
        "--years_test", type=int, default=2,
        help='Number of years (a year is 364 days) in the test dataset.'
    )
    parser.add_argument(
        "--calibration_window", type=int, default=4 * 364,
        help='Number of days used in the training dataset for recalibration.'
    )
    parser.add_argument(
        "--begin_test_date", type=str, default=None,
        help='Optional parameter to select the test dataset start date (format: d/m/Y H:M).'
    )
    parser.add_argument(
        "--end_test_date", type=str, default=None,
        help='Optional parameter to select the test dataset end date (format: d/m/Y H:M).'
    )
    return parser.parse_args()

def setup_logging():
    log_file_name = 'forecasting.log'
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def log_system_info():
    logging.info(f'Python Version: {sys.version}')
    logging.info(f'Platform: {sys.platform}')
    logging.info(f'Operating System: {os.name}')
    logging.info(f'Script Path: {os.path.abspath(__file__)}')
    logging.info(f'Execution Script Name: {os.path.basename(__file__)}')

    library_scripts = [
        'epftoolbox.data',
        'epftoolbox.evaluation',
        'epftoolbox.models'
    ]
    for lib_script in library_scripts:
        lib_script_name = os.path.basename(lib_script)
        lib_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), lib_script))
        logging.info(f'Library Script Name: {lib_script_name}')
        logging.info(f'Library Script Path: {lib_script_path}')

def log_function_info(func):
    def wrapper(*args, **kwargs):
        script_name = os.path.basename(__file__)
        func_name = func.__name__
        logging.info(f'{script_name} - {func_name} - Start')
        result = func(*args, **kwargs)
        logging.info(f'{script_name} - {func_name} - End')
        return result
    return wrapper

@log_function_info
def load_data(args):
    path_datasets_folder = os.path.join('.', 'datasets')
    return read_data(
        dataset=args.dataset,
        years_test=args.years_test,
        path=path_datasets_folder,
        begin_test_date=args.begin_test_date,
        end_test_date=args.end_test_date
    )

@log_function_info
def create_forecast_file_name(args):
    return f'fc_nl_dat{args.dataset}_YT{args.years_test}_CW{args.calibration_window}.csv'

@log_function_info
def create_forecast_file_path(args):
    path_recalibration_folder = os.path.join('.', 'experimental_files')
    return os.path.join(path_recalibration_folder, create_forecast_file_name(args))

@log_function_info
def process_date(args, df_train, df_test, forecast, real_values, model, date):
    script_name = os.path.basename(__file__)
    func_name = process_date.__name__
    logging.info(f'{script_name} - {func_name}: Processing date: {str(date)[:10]}')

    data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)
    data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN
    Yp = model.recalibrate_and_forecast_next_day(
        df=data_available,
        next_day_date=date,
        calibration_window=args.calibration_window
    )
    forecast.loc[date, :] = Yp
    mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values))
    smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100
    log_message = f'{str(date)[:10]} - sMAPE: {smape:.2f}%  |  MAE: {mae:.3f}'
    print(log_message)
    logging.info(log_message)

@log_function_info
def main():
    try:
        setup_logging()
        log_system_info()
        args = parse_arguments()

        logging.info(f'Starting the script for dataset: {args.dataset}')
        logging.info(f'Test years: {args.years_test}')
        logging.info(f'Calibration window: {args.calibration_window}')

        df_train, df_test = load_data(args)
        forecast_file_path = create_forecast_file_path(args)

        forecast = pd.DataFrame(index=df_test.index[::24], columns=[f'h{k}' for k in range(24)])
        real_values = df_test['Price'].values.reshape(-1, 24)
        real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

        forecast_dates = forecast.index
        model = LEAR(calibration_window=args.calibration_window)

        for date in forecast_dates:
            process_date(args, df_train, df_test, forecast, real_values, model, date)

        logging.info('Script execution completed successfully.')

    except Exception as e:
        handle_error(e)

@log_function_info
def handle_error(e):
    script_name = os.path.basename(__file__)
    func_name = handle_error.__name__
    error_message = f"An error occurred in {script_name} - {func_name}: {str(e)}"
    print(error_message)
    logging.error(error_message)
    
    # Include traceback in the log file
    logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
