import pandas as pd
import numpy as np
from copy import deepcopy
from gplearn.genetic import SymbolicTransformer
import re


class FeatureAugment:
    """
    Feature augmentation
    """
    @staticmethod
    def operations(df, features):
        copy_df = deepcopy(df)
        df_new = copy_df[features]
        df_new = df_new - df_new.min()
        sqr_name = [str(fa)+"_pow_2" for fa in df_new.columns]
        log_p_name = [str(fa)+"_log_p_abs" for fa in df_new.columns]
        rec_p_name = [str(fa)+"_inv_p" for fa in df_new.columns]
        sqrt_name = [str(fa)+"_sqrt_p" for fa in df_new.columns]

        df_sqr = pd.DataFrame(np.power(df_new.values, 2), columns=sqr_name, index=df.index)
        df_log = pd.DataFrame(np.log(df_new.add(1.0).abs().values), columns=log_p_name, index=df.index)
        df_rec = pd.DataFrame(np.reciprocal(df_new.add(1.0).values), columns=rec_p_name, index=df.index)
        df_sqrt = pd.DataFrame(np.sqrt(df_new.abs().add(1.0).values), columns=sqrt_name, index=df.index)

        dfs = [copy_df, df_sqr, df_log, df_rec, df_sqrt]
        return pd.concat(dfs, axis=1)

    @staticmethod
    def genetic_feat(df, target, num_gen=20, num_comp=10):
        """
        :param df:
        :param target:
        :param num_gen:
        :param num_comp:
        :return:
        """
        # function_set = ['add', 'sub', 'mul', 'div',
        #                 'sqrt', 'log', 'abs', 'neg', 'inv', 'tan']
        # function_set = ['add', 'sub', 'mul', 'div', "neg"]
        function_set = ['add', 'sub', "neg"]
        gp = SymbolicTransformer(generations=num_gen, population_size=200,
                                 hall_of_fame=100, n_components=num_comp,
                                 function_set=function_set,
                                 parsimony_coefficient=0.0005,
                                 max_samples=0.9, verbose=1,
                                 random_state=0, n_jobs=6, metric='pearson')
        drop_target_df = df.drop(target, axis=1)
        gen_feats = gp.fit_transform(drop_target_df, df[target])
        non_repeat_col_index = []
        set_ = set()
        for i in range(len(gp._best_programs)):
            val = str(gp._best_programs[i])
            if val in set_:
                continue
            else:
                non_repeat_col_index.append(i)
                set_.add(val)
        print(set_)
        print(non_repeat_col_index)
        col_lst = drop_target_df.columns
        gen_feats = pd.DataFrame(gen_feats[:, np.array(non_repeat_col_index)],
                                 columns=[FeatureAugment.parse_equation(col_lst, str(gp._best_programs[idx]))
                                                     for idx in non_repeat_col_index])
        for col in gen_feats.columns:
            if col in drop_target_df.columns:
                gen_feats = gen_feats.drop(col, axis=1)
        print("gen_feats.columns = ", gen_feats.columns)
        gen_feats.index = df.index
        return pd.concat((drop_target_df, gen_feats), axis=1)

    @staticmethod
    def parse_equation(col_lst, program_str):
        num_lst = map(int, re.findall("X([0-9]+)", program_str))
        for num in num_lst:
            replace_col = col_lst[num]
            program_str = program_str.replace("X" + str(num), replace_col)
        return program_str


class InstanceAugmentation:
    @staticmethod
    def add_randn_noise(data, row_ids, seed=None, fraction=0.05):
        if fraction < 0 or fraction > 1:
            raise RuntimeError("{} is out of range, fraction must be in range [0, 1]".format(fraction))
        if seed is None:
            seed = 1500
        random_state = np.random.RandomState(seed)
        new_data = np.zeros((len(row_ids), data.shape[1]))
        noise_std_arr = fraction * np.std(data, axis=0)
        for ii, n in enumerate(noise_std_arr):
            new_data[:, ii] = random_state.normal(loc=0, scale=n, size=len(row_ids))
        new_data = new_data + data.iloc[row_ids, :]
        return new_data


def tabular_feature_augmentation(csv_dir,
                                 drop_features_lst,
                                 target,
                                 save_dir,
                                 index_col=0,
                                 is_operation_augmentation=True,
                                 is_symbolic_transform=True
                                 ):
    """
    :param csv_dir: input csv file address
    :param drop_features_lst: no need to feature augmentation column
    :param target: target column
    :param save_dir: output csv file address
    :param index_col:
    :param is_operation_augmentation:
    :param is_symbolic_transform:
    :return:
    """
    origin_data_frame = pd.read_csv(csv_dir, index_col=index_col)
    aug_data_frame = origin_data_frame.drop(drop_features_lst + [target], axis=1)
    col_lst = list(aug_data_frame.columns)
    if is_operation_augmentation:
        aug_data_frame = FeatureAugment.operations(aug_data_frame, col_lst)
    if is_symbolic_transform:
        linshi_frame = pd.concat([origin_data_frame[[target]], aug_data_frame], axis=1, ignore_index=False)
        aug_data_frame = FeatureAugment.genetic_feat(linshi_frame, target=target)
    # print(aug_data_frame.columns)
    aug_data_frame = pd.concat([origin_data_frame[drop_features_lst + [target]], aug_data_frame],
                               axis=1)
    aug_data_frame.to_csv(save_dir)


if __name__ == "__main__":
    csv_dir = "../../data/en_pressure.csv"
    drop_features_lst = ["element"]
    target = "PNAS_en"
    tabular_feature_augmentation(csv_dir,
                                 drop_features_lst,
                                 target,
                                 save_dir="../../data/augment_en_pressure.csv",
                                 index_col=0,
                                 is_operation_augmentation=True,
                                 is_symbolic_transform=True,
                                 )
    # csv_dir = "../../data/en_pressure.csv"
    # target = "PNAS_en"
    # drop_features_lst = ["element", "en", "ionization_energy", "atomic_number", "pseudo_row"]
    # drop_features_lst = ["element"]
