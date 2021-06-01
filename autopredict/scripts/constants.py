# below config can be passed as input parameters
# for POC using hard coded
threads = 2
input_path = 'C:\\Users\\sanchit.latawa\\Desktop\\csv_parquet\\incoming_data\\'
process_path = 'C:\\Users\\sanchit.latawa\\Desktop\\csv_parquet\\processed\\'
config_path = 'C:\\Users\\sanchit.latawa\\Desktop\\csv_parquet\\config\\config.txt'
output_path = 'C:\\Users\\sanchit.latawa\\Desktop\\csv_parquet\\parquet\\'
tmout = 120
col_sep = '(\'~\')'
rec_delim = '#@#@#'
chunk_size = 18000
# constants
ready = 0
processing = 100
success = 200
error = 400
spool = True
file_tab = ''
idle_time_th_ss = 30
sleep_ss = 30