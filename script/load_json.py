# %% package import
import numpy as np
import pandas as pd
import time
import json
import ast
from os import listdir
from os.path import isfile, join
import natsort

# %%


def log_read(log_dir):
    #log_dir = 'logs_by_frame'

    log_files = [f for f in listdir(log_dir) if isfile(join(log_dir, f))]
    log_files = natsort.natsorted(log_files)

    log_dat = []

    for file_index in range(0, len(log_files)):

        file_path = join(log_dir, log_files[file_index])

        tmp_dat = open(file_path, 'r').read()
        tmp_dat = json.loads(tmp_dat)

        log_dat.append(tmp_dat)

    return log_dat

# %%
