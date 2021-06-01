from pyspark.sql import SparkSession
from scripts import utils
import os
import json
from scripts import constants
from glob import glob
import os
import time
# below config can be passed as input parameters
# for POC just passing these as constant hard coded


def convert_csv_parquet(files):
    print('---------- Start: convert_csv_parquet ----------------')
    start_time = time.time()
    print(f'Total files to be converted to pqt {len(files)}')
    for file in files:
        file = os.path.basename(file)
        print(f'Converting file - {file} from csv to pqt')
        sc = SparkSession.builder.master('local[*]') \
             .config("spark.driver.memory", "15g").appName('test').getOrCreate()
        print(file)
        df_csv = utils.read_csv(sc, constants.process_path + file)
        print(df_csv.count())
        pqt_file = file.replace('.txt','')
        utils.convertcsv_toparquet(df_csv, constants.output_path+pqt_file)
        os.rename(constants.process_path + file, constants.process_path + 'bk_' + file)
        print(f'Converting file complete- {file} from csv to pqt')
    print("--- %s seconds ---" % (time.time() - start_time))
    print('---------- Complete: convert_csv_parquet ----------------')


if __name__ == '__main__':
    # this program can be run in polling mode
    # so as tp be ready for any incoming event
    start_inactive_time = time.time()
    while 1:
        try:
            files = glob(constants.process_path + 'con*')
            if files:
                convert_csv_parquet(files)
                # reset timer
                start_inactive_time = time.time()
            else:
                time.sleep(constants.sleep_ss)

            if (time.time() - start_inactive_time) >= constants.idle_time_th_ss:
                print('Idle time Threshold Breached - exiting')
                break
        except:
            utils.print_exception('Main: csvtoparquet')

