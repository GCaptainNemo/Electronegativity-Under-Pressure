import sys
import os
from os.path import dirname
sys.path.insert(0, dirname(dirname(dirname(os.path.abspath(__file__)))))
from pysr import *
import pandas as pd
import numpy as np
from src.utils.plot_figure import plt_result
from copy import deepcopy


model = PySRRegressor(
        maxsize=25,
        niterations=5,
        binary_operators=["+", "*", "pow"],
        unary_operators=[
                             "exp",
                             # "sin",  # Pre-defined library of operators (see docs)
                             # "inv(x) = 1/x",  # Define your own operator! (Julia syntax)
                             # "sqrt",
                             # "square",
                             # "log"
                         ],
        # extra_sympy_mappings={'inv': lambda x: 1 / x},
        # extra_torch_mappings={sympy.core.numbers.Rational: torch.FloatTensor,
        #                       sympy.core.numbers.Half: torch.FloatTensor},
        model_selection="accuracy",   # model_selection="best", # OR "accuracy",
        # loss="loss(x, y) = abs(x - y)",  # Custom loss function (julia syntax)
        constraints={'pow': (1, 1), 'mult': (3, 3)},
        annealing=False,
        useFrequency=False,
        optimizer_algorithm="BFGS",
        optimizer_iterations=10,
        optimize_probability=1,
        # tempdir=self.save_dir + "formulas/",
        # verbosity=1,
        temp_equation_file=True,  # 是否生成临时文件
        delete_tempfiles=False,  # 是否最后删除临时文件（不能选True）
    )

s_block_metal = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr",
                 "Cs", "Ba", "Fr", "Ra"]
ds_block_metal = ["Cu", "Zn", "Ag", "Cd", "Au", "Hg"]
p_block_metal = ["Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi", "Po"]
d_block_metal = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
                 "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                 "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt"]
f_block_metal = [
                  "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                  "Tb", "Dy", "Ho", "Er", "Tm", "Yb",  "Ac", "Th",
                  "Pa", "U", "Np", "Pu", "Am", "Cm",
                  ]
noble_gas = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
non_metal = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I", "At"]
df = pd.read_csv("../../data/augment_en_pressure.csv", index_col=0)
# df = pd.read_csv("../../data/interpolation_sr_df.csv")

s_block_metal_df = df[df["element"].isin(s_block_metal)]
p_block_metal_df = df[df["element"].isin(p_block_metal)]
d_block_metal_df = df[df["element"].isin(d_block_metal)]
f_block_metal_df = df[df["element"].isin(f_block_metal)]
ds_block_metal_df = df[df["element"].isin(ds_block_metal)]

noble_gas_df = df[df["element"].isin(noble_gas)]
non_metal_df = df[df["element"].isin(non_metal)]
input_column_tokens = ["atom_s", "atom_p", "atom_d", "atom_f",
                       "row", "pressure", "atomic_number", "group"]
# input_column_tokens = ["atom_s", "atom_p", "atom_d", "atom_f",
#                        "row", "pressure", "group"]
# print(noble_gas_df)


def fitting_df(df, noise_intensity=1e-3, is_remove_equal=True):
    y = df[["PNAS_en"]].to_numpy(dtype=np.float32)
    y = np.concatenate([y, y], axis=0)
    cpy_input_column_tokens = deepcopy(input_column_tokens)
    feature_df = df[cpy_input_column_tokens]
    std_df = feature_df.std()
    for i in range(len(cpy_input_column_tokens) - 1, -1, -1):
        col = cpy_input_column_tokens[i]
        if std_df[col] < 1e-3:
            cpy_input_column_tokens.pop(i)
            print(col, ": constant")
        else:
            print("{} std:".format(col), std_df[col])
    print("cpy_input_column_tokens = ", cpy_input_column_tokens)
    feature_df = df[cpy_input_column_tokens] # rm constant feature
    x = feature_df.to_numpy(dtype=np.float32)
    std_ = np.std(x, axis=0, keepdims=True)
    x = np.concatenate([x, x + np.random.randn(*x.shape) * std_ * noise_intensity])
    model.fit(x, y)
    chose_sympy_expression = model.sympy()
    print(chose_sympy_expression)


def start_fitting_all(is_noble_gas=True, is_non_metal=True,
                      is_s_block=True, is_p_block=True,
                      is_d_block=True, is_f_block=True, is_ds_block=True):
    if is_noble_gas:
        print("==============noble gas ==============")
        fitting_df(noble_gas_df)
    if is_non_metal:
        print("==============non_metal ==============")
        fitting_df(non_metal_df)
    if is_s_block:
        print("==============s_block ==============")
        fitting_df(s_block_metal_df)
    if is_p_block:
        print("==============p_block ==============")
        fitting_df(p_block_metal_df)
    if is_d_block:
        print("==============d_block ==============")
        fitting_df(d_block_metal_df)
    if is_f_block:
        print("==============f_block ==============")
        fitting_df(f_block_metal_df)
    if is_ds_block:
        print("==============ds_block ==============")
        fitting_df(ds_block_metal_df)


def corrected_coefficient(df, func):
    gt_y = df[["PNAS_en"]].to_numpy(dtype=np.float32)
    x = df[input_column_tokens].to_numpy(dtype=np.float32)
    pred_y = func(x).reshape(-1, 1)
    A = np.concatenate([pred_y, np.ones_like(pred_y, dtype=np.float32)], axis=1)
    x = np.linalg.pinv(A) @ gt_y
    return x[0, 0], x[1, 0]


def predicted_plot(df, func, title="", write_csv=True, criterion_type="mse"):
    gt_y = df[["PNAS_en"]].to_numpy(dtype=np.float32)
    x = df[input_column_tokens].to_numpy(dtype=np.float32)
    print(np.min(x[:, 2]))
    pred_y = func(x).reshape(-1, 1)
    print(pred_y.shape)
    print(x.shape, gt_y.shape)
    if write_csv:
        dict_ = {"pred_y": [], "gt_y": []}
        dict_["pred_y"] = list(pred_y.reshape(-1))
        dict_["gt_y"] = list(gt_y.reshape(-1))
        save_df = pd.DataFrame(dict_)
        save_df.to_csv("{}.csv".format(title))
    mse = np.mean((pred_y - gt_y) ** 2)
    if criterion_type == "mse":
        title += " MSE: {:.3f}".format(mse)
    else:
        title += " RMSE: {:.3f}".format(mse ** 0.5)
    plt_result(predict_data=pred_y, gt_data=gt_y, title=title)


if __name__ == "__main__":
    start_fitting_all(is_noble_gas=False, is_non_metal=True,
                      is_s_block=False, is_p_block=False,
                      is_d_block=False, is_f_block=False, is_ds_block=False)
