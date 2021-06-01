from multiprocessing.dummy import Pool as ThreadPool
import os
import time
from scripts import utils
import json
from filelock import FileLock, Timeout
from scripts import constants
from glob import glob


def create_files_metadata_config(input_path, config_path):
    meta = {}
    if not os.path.exists(config_path):
        # config file does not exist
        for rec in glob(constants.input_path + 'con*'):
            rec = os.path.basename(rec)
            meta[rec] = 0

        utils.write_file(config_path, meta)
    print('write file success')


def get_files_list(input_path, config_path):
    cnt = 0
    file_to_be_migrated = []
    # as input files are static the below function is
    # candidate to be moved out to the calling Job/shell script
    create_files_metadata_config(input_path, config_path)
    # lock config file
    with FileLock(config_path + '.lock', timeout=constants.tmout):
        # fetch config metadata
        metadata = utils.read_file(config_path)
        for key, value in metadata.items():
            if cnt >= constants.threads:
                break
            if value == constants.ready:
                cnt += 1
                file_to_be_migrated.append(key)
                metadata[key] = 100
        utils.write_file(config_path, metadata)

    return file_to_be_migrated, [(input_path + file, constants.process_path + file, constants.col_sep, constants.rec_delim, constants.chunk_size) for file in file_to_be_migrated]


def update_metadata(file_to_be_migrated, status):
    with FileLock(constants.config_path + '.lock', timeout=constants.tmout):
        metadata = utils.read_file(constants.config_path)
        for rec in file_to_be_migrated:
            metadata[rec] = status
        utils.write_file(constants.config_path, metadata)


if __name__ == '__main__':
    # create config file to maintain status of
    # file pre-process this is required as
    # a centralized metadata of raw files which would be used by nodes
    # to understand the current status of files
    try:
        print('---------- Started: Pre-Processing ----------------')
        start_time = time.time()
        file_to_be_migrated, file_tab = get_files_list(constants.input_path, constants.config_path)
        print(file_tab)
        while file_tab:
            start_time = time.time()
            pool = ThreadPool(int(constants.threads))
            pool.starmap(utils.massage_data, file_tab)
            pool.close()
            pool.join()
            update_metadata(file_to_be_migrated, constants.success)
            # reload file tab
            file_to_be_migrated, file_tab = get_files_list(constants.input_path, constants.config_path)
        else:
            print("--- %s seconds ---" % (time.time() - start_time))
            print('---------- Completed: Pre-Processing ----------------')
    except:
        # error out the files at hand
        if file_tab:
            update_metadata(file_to_be_migrated, constants.error)
        utils.print_exception('Main: Pre-Processing')
