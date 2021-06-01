from scripts import utils
from scripts import csvtoparquetconverter
from scripts import preprocess_data
from scripts import constants
import os

## Usage from terminal
## pytest -v ./tests

def test_dummy_file_creation():
    sample_data = "1012001'~'CEC2017KOSU0053'~'2018-01-13 00:00:00'~'W.P.TRADING CORP'~''~'350 5th Avenue'~'New York'~'New York'~'10118'~'United States'~'44425805'~'KING OCEAN AGENCY INC.'~'Parque Industrial Machangara, Panamericana Norte km 4, Cuenca, Azuay, Ecuador'~''~'Cuenca'~'Azuay'~''~'Ecuador'~'33499873'~'GRAIMAN CIA. LTDA.'~'0190122271001'~''~'A Customs Brokerage, Inc.'~'GUAYAQUIL'~'PORT EVERGLADES'~'United States'~'UNITED STATES OF AMERICA'~''~''~''~''~''~''~'PLANET V'~'Antigua And Barbuda'~'AG'~'PORCELANA'~'31'~'PAQUETES'~'26770'~'STC GLAZED PORCELAIN'~'1'~'0'~''~'KING OCEAN ECUADOR KINGOCEAN S.A.'~'2018-03-20 00:00:00#@#@#"
    output_file = 'test_file'
    row = 1000
    result = utils.create_dummy_file(sample_data, rows=row, output_file=output_file)
    # clean up
    os.remove(output_file)
    assert result


def read_write_file():
    metadata = {'test':1}
    config_path ='test_read_write_function'
    utils.write_file(config_path, metadata)
    w_ind = os.path.exists(config_path)
    data = utils.read_file(config_path)
    os.remove(config_path)
    w_ind = 'False'
    assert w_ind and data == metadata


def test_masssage_data():
    sample_data = "1012001'~'CEC2017KOSU0053'~'2018-01-13 00:00:00'~'W.P.TRADING CORP'~''~'350 5th Avenue'~'New York'~'New York'~'10118'~'United States'~'44425805'~'KING OCEAN AGENCY INC.'~'Parque Industrial Machangara, Panamericana Norte km 4, Cuenca, Azuay, Ecuador'~''~'Cuenca'~'Azuay'~''~'Ecuador'~'33499873'~'GRAIMAN CIA. LTDA.'~'0190122271001'~''~'A Customs Brokerage, Inc.'~'GUAYAQUIL'~'PORT EVERGLADES'~'United States'~'UNITED STATES OF AMERICA'~''~''~''~''~''~''~'PLANET V'~'Antigua And Barbuda'~'AG'~'PORCELANA'~'31'~'PAQUETES'~'26770'~'STC GLAZED PORCELAIN'~'1'~'0'~''~'KING OCEAN ECUADOR KINGOCEAN S.A.'~'2018-03-20 00:00:00#@#@#"
    raw_file = 'test_file'
    csv_file = 'csv_file'
    rec_sep = constants.col_sep
    line_sep = constants.rec_delim
    chunk_size = constants.chunk_size
    row = 1000
    result = utils.create_dummy_file(sample_data, rows=row, output_file=raw_file)
    if result:
        assert utils.massage_data(raw_file, csv_file, rec_sep, line_sep, chunk_size)
        # clean-up
        os.remove(raw_file)
        os.remove(csv_file)
    else:
        assert False

#
# if __name__ == '__main__':
#     #test_dummy_file_creation()
#     #read_write_file()
#     test_masssage_data()