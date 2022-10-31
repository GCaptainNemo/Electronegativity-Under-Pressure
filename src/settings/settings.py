import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.neural_net.criterion import MSELoss, HuberLoss
from src.neural_net.normalize import NormalizeDataframeObj
import os


class TaskSettingObj:
    """
    setting task parameters:
    input columns, predict column, separate dataset
    """
    is_data_aug = True
    x_data_aug_noise_std = 0.01
    y_data_aug_noise_std = 0.002
    input_csv_dir = "../../data/augment_en_pressure.csv"
    drop_features_lst = ["element"]
    target = "PNAS_en"
    result_output_dir = "../../src/neural_net/training_result"
    if not os.path.exists(result_output_dir):
        os.mkdir(result_output_dir)
    # split dataset
    origin_full_df = pd.read_csv(input_csv_dir, index_col=0).drop(drop_features_lst, axis=1)
    input_column_names = list(origin_full_df.columns)
    input_column_names.remove(target)
    if np.any(np.isnan(origin_full_df.to_numpy().astype(np.float32))):
        print("[error] NaN in origin data frame!!!!!!")
        raise ValueError("[error] NaN in origin data frame!!!!!!")
    # normalized label for better convergence performance
    is_normalize_y = True
    is_normalize_x = True
    random_seed = 2
    random_state = np.random.RandomState(random_seed)
    origin_train_val_df, origin_test_df = train_test_split(
        origin_full_df, random_state=random_state, test_size=.1)
    # normalize only use train set and val set
    normalize_dataframe_obj = NormalizeDataframeObj(result_output_dir + "/pre_processing")
    normalize_full_df = origin_full_df.copy()
    if is_normalize_y:
        y_mu, y_sigma = normalize_dataframe_obj.normalize_features(df=origin_train_val_df,
                                                                   col_name_lst=[target],
                                                                   option="z_score",
                                                                   return_type="par")
        normalize_full_df[[target]] = (normalize_full_df[[target]] - y_mu) / y_sigma
    if is_normalize_x:
        x_mu, x_sigma = normalize_dataframe_obj.normalize_features(df=origin_train_val_df,
                                                                   col_name_lst=input_column_names,
                                                                   option="z_score",
                                                                   return_type="par")
        normalize_full_df[input_column_names] = (normalize_full_df[input_column_names] - x_mu) / x_sigma
    # split again
    if np.any(np.isnan(normalize_full_df.to_numpy())):
        print("[error] NaN in normalize data frame!!!!!!")
        raise ValueError("[error] NaN in normalize data frame!!!!!!")
    random_seed = 2
    random_state = np.random.RandomState(random_seed)
    train_val_df, test_df = train_test_split(
        normalize_full_df, random_state=random_state, test_size=.1)
    train_criterion = HuberLoss()
    test_criterion = MSELoss()
    validation_batch_size = 500
    layer_num = 3
    # baseline train parameters
    baseline_is_use_attention = True
    baseline_save_dir = result_output_dir + "/baseline_model/"
    baseline_batch_size = 64
    baseline_base_learner_num = 5
    baseline_lr = 19   # 19              # 19 * e-4
    baseline_step = 15
    baseline_epoch = 750
    baseline_add_l1 = False
    baseline_l1_alpha = 1e-8
