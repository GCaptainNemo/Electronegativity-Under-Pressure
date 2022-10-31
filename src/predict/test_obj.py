import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
import torch
from src.settings.settings import TaskSettingObj
import pandas as pd
from copy import deepcopy
from src.neural_net.predict import get_loss, get_predict
import numpy as np
from src.predict.utils import *
from pymatgen.core import Composition, Element
from src.utils.plot_figure import plt_result
from collections import defaultdict
import matplotlib.pyplot as plt


class TestObj:
    def __init__(self,
                 model_save_address="../neural_net/training_result/baseline_model/ensemble_baseline.pth"):
        self.baseline_model = torch.load(model_save_address, map_location=torch.device('cpu'))
        self.baseline_model.eval()
        self.y_mu, self.y_sigma = float(TaskSettingObj.y_mu), float(TaskSettingObj.y_sigma)
        self.normalize_element_property_dict = dict()
        self.origin_element_property_dict = dict()  # element: property
        self.get_each_element_property()
        self.pressure_mu_sigma_dict = {}
        self.get_pressure_info()
        self.pressure_lambda_dict = {
            "pressure": lambda pressure: pressure,
            "pressure_pow_2": lambda pressure: pressure ** 2,
            "pressure_log_p_abs": lambda pressure: np.log(np.abs(pressure + 1.0)),
            "pressure_inv_p": lambda pressure: np.reciprocal(pressure + 1.0),
            "pressure_sqrt_p": lambda pressure: np.sqrt(np.abs(pressure) + 1.0),
        }
        # self.predict("He", 50)

    def get_each_element_property(self):
        """
        get element property
        :return:
        """
        element_array = pd.read_csv(TaskSettingObj.input_csv_dir, index_col=0)[["element"]]
        normalize_property_array = TaskSettingObj.normalize_full_df[TaskSettingObj.input_column_names]
        origin_property_array = TaskSettingObj.origin_full_df[TaskSettingObj.input_column_names]
        for i in range(element_array.shape[0]):
            element = str(element_array.iloc[i, 0])
            if element not in self.normalize_element_property_dict:
                # normalized
                self.normalize_element_property_dict[element] = normalize_property_array.iloc[[i]]
            if element not in self.origin_element_property_dict:
                # Not normalized
                self.origin_element_property_dict[element] = origin_property_array.iloc[[i]]

    def get_pressure_info(self):
        self.include_pressure_col_lst = []
        for col in TaskSettingObj.normalize_full_df.columns:
            if "pressure" in col:
                self.include_pressure_col_lst.append(col)
        pressure_mu, pressure_sigma = TaskSettingObj.normalize_dataframe_obj.normalize_features(
            df=TaskSettingObj.origin_train_val_df,
            col_name_lst=self.include_pressure_col_lst,
            option="z_score",
            return_type="par")
        for i, key in enumerate(self.include_pressure_col_lst):
            self.pressure_mu_sigma_dict[key] = [pressure_mu[i], pressure_sigma[i]]

    def predict_formula_pressure(self, formula, input_pressure):
        """
        formula: e.g., H2O
        input_pressure: Gpa
        """
        element_dict = Composition(formula, strict=True).as_dict()
        sum_en = 0.0
        sum_num = sum(val for val in element_dict.values())
        for key, val in element_dict.items():
            en = self.predict_element_pressure(key, input_pressure=input_pressure)
            sum_en += en * val
        sum_en /= sum_num
        print(formula, "at {}Gpa pred_en:".format(input_pressure), sum_en)
        return sum_en

    def predict_formula_pressures(self, formula, start_pressure, end_pressure, step):
        sum_en_lst = []
        for pressure in np.arange(start_pressure, end_pressure, step):
            en = self.predict_formula_pressure(formula, pressure)
            sum_en_lst.append(en)
        print(formula, "at [{}, {}) Gpa (step = {}): \n".format(start_pressure, end_pressure, step))
        print(sum_en_lst)
        return sum_en_lst

    def predict_element_pressures(self, element_name, start_pressure, end_pressure, step):
        en_lst = []
        for pressure in np.arange(start_pressure, end_pressure, step):
            en = self.predict_element_pressure(element_name, pressure)
            en_lst.append(en)
        print(element_name, "at [{}, {}) Gpa (step = {}): \n".format(start_pressure, end_pressure, step))
        print(en_lst)
        return en_lst

    def predict_element_pressure(self, element_name, input_pressure, is_for_sr=False):
        # Calculate the value of the feature related to pressure
        normalize_input_pd_frame = deepcopy(self.normalize_element_property_dict[element_name])
        origin_input_pd_frame = deepcopy(self.origin_element_property_dict[element_name])
        for key, val in self.pressure_lambda_dict.items():
            origin_value = self.pressure_lambda_dict[key](input_pressure)
            origin_input_pd_frame[key] = origin_value
            mu, sigma = self.pressure_mu_sigma_dict[key][0], self.pressure_mu_sigma_dict[key][1]
            normalize_input_pd_frame[key] = (origin_value - mu) / sigma
        # calculate pressure-correlated feature(Symbolic Transformer augmented features)
        for pressure_key in self.include_pressure_col_lst:
            if pressure_key in self.pressure_lambda_dict:
                continue
            prefix_token_lst = lexical_analysis(pressure_key)
            origin_res = calculate(prefix_token_lst, origin_input_pd_frame)
            mu, sigma = self.pressure_mu_sigma_dict[pressure_key][0], \
                        self.pressure_mu_sigma_dict[pressure_key][1]
            normalize_input_pd_frame[pressure_key] = (origin_res - mu) / sigma
            origin_input_pd_frame[pressure_key] = origin_res
        input_x_array = torch.from_numpy(normalize_input_pd_frame.to_numpy()).reshape(1, -1)
        input_x_array = input_x_array.to(torch.float32)
        with torch.no_grad():
            pred_y = self.baseline_model(input_x_array)
            pred_en = float(pred_y.item()) * self.y_sigma + self.y_mu
            print(element_name, "at {}Gpa pred_en:".format(input_pressure), pred_en)
        if is_for_sr:
            return pred_en, origin_input_pd_frame
        else:
            return pred_en

    def debug_get_loss(self, option="test"):
        if option == "test":
            pred_y_array, gt_y = get_predict(
                model=self.baseline_model,
                test_dataframe=TaskSettingObj.test_df,
                col_names_lst=TaskSettingObj.input_column_names,
                target=TaskSettingObj.target,
                is_cuda=False
            )
        elif option == "full":
            pred_y_array, gt_y = get_predict(
                model=self.baseline_model,
                test_dataframe=TaskSettingObj.normalize_full_df,
                col_names_lst=TaskSettingObj.input_column_names,
                target=TaskSettingObj.target,
                is_cuda=False
            )
        pred_y_array = pred_y_array * self.y_sigma + self.y_mu
        gt_y = gt_y * self.y_sigma + self.y_mu
        print(pred_y_array.shape, gt_y.shape, type(pred_y_array))
        RMSE = (torch.mean((pred_y_array - gt_y) ** 2, axis=0) ** 0.5).item()
        MAE = torch.mean(np.abs(pred_y_array - gt_y)).item()
        R2 = np.corrcoef(pred_y_array.T, gt_y.T)[0, 1] ** 2
        print("RMSE = ", RMSE)
        print("MAE = ", MAE)
        print("R2 = ", R2)
        plt_result(
            predict_data=pred_y_array.numpy(),
            gt_data=gt_y.numpy(),
            title="DAN {} RMSE = {:.3f}, MAE = {:.3f}, R2 = {:.3f}".format(option, RMSE, MAE, R2)
        )
        dict_ = defaultdict(str)
        dict_["gt_y"] = gt_y.reshape(-1, ).tolist()
        dict_["pred_y"] = pred_y_array.reshape(-1, ).tolist()
        df = pd.DataFrame(dict_)
        df.to_csv("./predict_gt_{}.csv".format(option), index=False)
        return pred_y_array, gt_y

    def calculate_minerals_df(self,
                              df_address="../../data/minerals_A_B.csv",
                              pressures_lst=[0, 350],
                              save_df_address="../../data/minerals_en.csv"):
        """
        calculate minerals work function and generate dataframe
        """
        minerals_df = pd.read_csv(df_address, encoding='utf-8')
        minerals_lst = minerals_df["minerals"].tolist()
        new_dict = {"minerals": minerals_lst}
        REE_replace = "((LaCePrNdPmSmEuGdTbDyHoErTmYbLuScY)0.058823529411764705)"
        LN_replace = "((LaCePrNdPmSmEuGdTbDyHoErTmYbLu)0.06666666666666667)"
        for pressure in pressures_lst:
            assert isinstance(pressure, int) or isinstance(pressure, float)
            en_lst = []
            for mineral_formula in minerals_lst:
                if "REE" in mineral_formula:
                    mineral_formula = mineral_formula.replace("REE", REE_replace)
                if "Ln" in mineral_formula:
                    mineral_formula = mineral_formula.replace("Ln", LN_replace)
                try:
                    electronegativity = self.predict_formula_pressure(mineral_formula, pressure)
                except Exception as e:
                    print("error", mineral_formula)
                    electronegativity = float("nan")
                en_lst.append(electronegativity)
            new_dict[str(pressure)] = en_lst
        mineral_pressure_df = pd.DataFrame(new_dict)
        mineral_pressure_df.to_csv(save_df_address)

    def en_interpolation(self, element_lst=["H"], obj_pressures=[0, 50, 200, 500]):
        plt.figure(0)
        for element in element_lst:
            gt_en_lst = []
            for pressure in obj_pressures:
                output_en = self.predict_element_pressure(element, pressure)
                gt_en_lst.append([pressure, output_en])
            interpolation_en_lst = []
            for pressure in range(0, 500):
                output_en = self.predict_element_pressure(element, pressure)
                interpolation_en_lst.append([pressure, output_en])
            plt.plot([x[0] for x in interpolation_en_lst], [x[1] for x in interpolation_en_lst])
            plt.scatter(x=[x[0] for x in gt_en_lst], y=[x[1] for x in gt_en_lst], c="r")
        plt.show()

    def interpolation_for_symbolic_regression(self, low=0, high=500, step=5, save_path="./interpolation_sr_df.csv"):
        concat_lst = []
        for i in range(1, 97):
            element_name = str(Element.from_Z(i))
            for pressure in np.arange(low, high, step):
                pred_en, origin_input_pd_frame = self.predict_element_pressure(element_name, pressure, is_for_sr=True)
                origin_input_pd_frame.insert(0, "element", element_name, allow_duplicates=False)
                origin_input_pd_frame.insert(1, "PNAS_en", pred_en, allow_duplicates=False)
                concat_lst.append(origin_input_pd_frame)
                # print(origin_input_pd_frame["pressure"], len(origin_input_pd_frame), type(origin_input_pd_frame))
        concat_df = pd.concat(concat_lst, axis=0)
        concat_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    assert len(sys.argv) == 4
    if sys.argv[3] == "e":
        test_obj = TestObj()
        test_obj.predict_element_pressure(sys.argv[1], float(sys.argv[2]))
    elif sys.argv[3] == "w":
        test_obj = TestObj()
        test_obj.predict_formula_pressure(sys.argv[1], float(sys.argv[2]))
    else:
        raise ValueError("[error] python test_obj.py H 0 e (electronegativity) or "
                         "python test_obj.py Mg1.6Fe0.4SiO4 0 w (work function)")
