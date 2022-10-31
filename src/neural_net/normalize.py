import pandas as pd
import numpy as np
from src.utils.get_logger import get_logger
import os


class NormalizeDataframeObj:
    def __init__(self, logger_dir="./"):
        if not os.path.exists(logger_dir):
            os.mkdir(logger_dir)
        logger_name = logger_dir + "/pre_processing.log"
        self.logger = get_logger(name="pre_processing_logger", filename=logger_name)

    def normalize_features(self,
                           df,
                           col_name_lst,
                           option="z_score", return_type="df"):
        """
        normalize dataframe features
        standard normalization: x = (x - mu) / sigma
        min-max: x = (x - min_x) / (max_x - min_x)
        """
        new_df = df[col_name_lst].copy(deep=True)
        if option == "z_score":
            mean = np.mean(new_df, axis=0)
            sigma = np.std(new_df, axis=0)
            sigma[sigma < 1e-2] = 1
            new_df = (new_df - mean) / sigma
            for i, _ in enumerate(col_name_lst):
                self.logger.info("{}: (mu, sigma) = ({}, {})".format(_, float(mean[i]), float(sigma[i])))
        else:
            raise ValueError("[Error] option not in [min-max, z-score]!!!")
        if return_type == "df":
            return new_df
        elif return_type == "par":
            return mean, sigma
        else:
            raise ValueError("[Error] option not in [df, par]!!!")
