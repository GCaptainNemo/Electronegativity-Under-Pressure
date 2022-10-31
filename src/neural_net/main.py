from src.neural_net.model import MultiBranchModel
from src.settings.settings import TaskSettingObj
import torch
from src.neural_net.train import Train
from src.neural_net.data import CustomDataset
from sklearn.model_selection import KFold, train_test_split
import os
from copy import deepcopy
from src.utils.get_logger import get_logger
from src.utils.plot_figure import plt_result
from src.neural_net.predict import load_k_fold_bagging_model, get_loss, get_predict
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np
import pandas as pd


class TrainBaselineObj:
    def __init__(
            self,
            save_dir=TaskSettingObj.baseline_save_dir,
            lr=TaskSettingObj.baseline_lr,
            step=TaskSettingObj.baseline_step,
            epoch_num=TaskSettingObj.baseline_epoch,
            is_add_l1=TaskSettingObj.baseline_add_l1,
            l1_alpha=TaskSettingObj.baseline_l1_alpha,
            batch_size=TaskSettingObj.baseline_batch_size,
            base_learner_num=TaskSettingObj.baseline_base_learner_num
    ):

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self.lr = lr
        self.step = step
        self.epoch_num = epoch_num
        self.is_add_l1 = is_add_l1
        self.l1_alpha = l1_alpha
        self.batch_size = batch_size
        self.base_learner_num = base_learner_num
        self.logger = get_logger(filename=self.save_dir + "baseline_output.log",
                                 name="baseline_logger")

    def start_train(self):
        self.logger.info("====================================================")
        self.logger.info("Start Training Baseline")
        self.logger.info("====================================================")
        self.logger.info("Training Info: lr = {}, network_step = {}, epoch = {}".format(
            self.lr,
            self.step,
            self.epoch_num))
        self.logger.info("train criterion: {}, test criterion: {}".format(
            str(TaskSettingObj.train_criterion),
            str(TaskSettingObj.test_criterion)))

        col_group_lst = deepcopy([TaskSettingObj.input_column_names])
        target = TaskSettingObj.target
        prefix = self.save_dir + "baseline"
        train_branch_model_k_fold_bagging(
            col_group_lst=col_group_lst,
            tgt=target,
            prefix=prefix,
            criterion=TaskSettingObj.train_criterion,
            base_learner_num=self.base_learner_num,
            lr=self.lr,
            step=self.step,
            batch_size=self.batch_size,
            epoch_num=self.epoch_num,
            is_add_l1=self.is_add_l1,
            l1_alpha=self.l1_alpha,
            layer_num=TaskSettingObj.layer_num
        )

    def save_ensemble_model(self, name="ensemble_baseline.pth", batch_size=TaskSettingObj.validation_batch_size):
        multi_branch_model = load_k_fold_bagging_model(
            prefix=self.save_dir + "baseline",
            base_learner_num=TaskSettingObj.baseline_base_learner_num)
        multi_branch_model.cpu()
        full_df_loss = get_loss(
            model=multi_branch_model,
            test_dataframe=TaskSettingObj.normalize_full_df,
            col_names_lst=TaskSettingObj.input_column_names,
            target=TaskSettingObj.target,
            loss_func=TaskSettingObj.test_criterion,
            batch_size=batch_size
        )
        test_df_loss = get_loss(
            model=multi_branch_model,
            test_dataframe=TaskSettingObj.test_df,
            col_names_lst=TaskSettingObj.input_column_names,
            target=TaskSettingObj.target,
            loss_func=TaskSettingObj.test_criterion,
            batch_size=batch_size
        )
        self.logger.info("baseline full_df_loss({}) = {}".format(
            str(TaskSettingObj.test_criterion),
            full_df_loss))
        self.logger.info("baseline test_df_loss({}) = {}".format(
            str(TaskSettingObj.test_criterion),
            test_df_loss))
        with open(self.save_dir + name, "wb") as f:
            torch.save(multi_branch_model, f)
        self.logger.info("save {} successfully!!!".format(name))

    @staticmethod
    def test_baseline_model(
            save_dir=TaskSettingObj.baseline_save_dir,
            model_name="ensemble_baseline.pth",
            batch_size=TaskSettingObj.validation_batch_size
        ):
        with open(save_dir + model_name, "rb") as f:
            multi_branch_model = torch.load(f)
        loss = get_loss(
            model=multi_branch_model,
            test_dataframe=TaskSettingObj.normalize_full_df,
            col_names_lst=TaskSettingObj.input_column_names,
            target=TaskSettingObj.target,
            loss_func=TaskSettingObj.test_criterion,
            batch_size=batch_size
        )
        pred_y_array, gt_y = get_predict(
            model=multi_branch_model,
            test_dataframe=TaskSettingObj.normalize_full_df,
            col_names_lst=TaskSettingObj.input_column_names,
            target=TaskSettingObj.target,
        )
        print("data num = ", pred_y_array.shape[0], gt_y.shape[0])
        plt_result(
            predict_data=pred_y_array.numpy(),
            gt_data=gt_y.numpy(),
            title="Baseline model loss({}) = {}".format(str(TaskSettingObj.test_criterion),
                                                        loss.item())
        )


def train_branch_model_k_fold_bagging(
        col_group_lst,
        tgt,
        prefix,
        criterion,
        batch_size=64,
        is_add_l1=False,
        l1_alpha=0,
        base_learner_num=5,
        lr=19,
        step=15,
        epoch_num=750,
        layer_num=TaskSettingObj.layer_num
        ):
        """
        k fold bagging technique:
        """
        is_using_tabnet = False
        if base_learner_num >= 2:
            kf = KFold(n_splits=base_learner_num)
            fold_idx = 0
            # split train_val_df to get k-fold train set and validation set
            col_names = [_ for col in col_group_lst for _ in col]
            pred_y_lst = []
            for train_idx, val_idx in kf.split(TaskSettingObj.train_val_df):
                print("============fold_idx {}/{}===========".format(fold_idx, base_learner_num))
                train_df, val_df = TaskSettingObj.train_val_df.iloc[train_idx], TaskSettingObj.train_val_df.iloc[val_idx]
                print(len(train_df), len(val_df))
                train_dataset = CustomDataset(train_df, col_names, tgt)
                val_dataset = CustomDataset(val_df, col_names, tgt)
                if is_using_tabnet:
                    clf = TabNetRegressor()
                    clf.fit(train_dataset.X_tensor.cpu().numpy(), train_dataset.Y_tensor.cpu().numpy(),
                            eval_set=[(val_dataset.X_tensor.cpu().numpy(), val_dataset.Y_tensor.cpu().numpy())])
                    x_array = TaskSettingObj.test_df.drop([TaskSettingObj.target], axis=1).to_numpy()
                    gt_y_array = TaskSettingObj.test_df[[TaskSettingObj.target]].to_numpy()
                    pred_y_array = clf.predict(x_array)

                    pred_y_lst.append(pred_y_array)
                    continue
                model = MultiBranchModel(
                    input_dim_lst=[len(_) for _ in col_group_lst],
                    first_dim=50 + step,
                    second_dim=20 + step,
                    layer_num=layer_num
                ).cuda()

                train = Train(
                    model=model,
                    criterion=criterion,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    l1_alpha=l1_alpha,
                    batch_size=batch_size
                )
                train.train(epoch=epoch_num, lr=1e-4 * lr, is_add_l1=is_add_l1)
                with open(prefix + str(fold_idx) + ".pth", "wb+") as f:
                    torch.save(model, f)
                fold_idx += 1
            pred_y_array = np.mean(np.concatenate(pred_y_lst, axis=1), axis=1, keepdims=True)
            print(pred_y_array.shape, gt_y_array.shape)
            print("TABNET MSE = ", np.mean((pred_y_array - gt_y_array) ** 2))
            print("TABNET MAE = ", np.mean(np.abs(pred_y_array - gt_y_array)))
            print("TABNET R2 = ", np.corrcoef(pred_y_array.T, gt_y_array.T)[0, 1] ** 2)
        else:
            # base_learner_num = 1
            random_seed = 2
            random_state = np.random.RandomState(random_seed)
            origin_train_df, origin_val_df = train_test_split(
                TaskSettingObj.train_val_df, random_state=random_state, test_size=.25)
            col_names = [_ for col in col_group_lst for _ in col]
            print("============base_learner_idx 1/1===========")
            print(len(origin_train_df), len(origin_val_df))
            # print(origin_train_df[tgt].iloc[0])
            train_dataset = CustomDataset(origin_train_df, col_names, tgt)
            val_dataset = CustomDataset(origin_val_df, col_names, tgt)
            if is_using_tabnet:
                clf = TabNetRegressor()
                clf.fit(train_dataset.X_tensor.cpu().numpy(), train_dataset.Y_tensor.cpu().numpy(),
                        eval_set=[(val_dataset.X_tensor.cpu().numpy(), val_dataset.Y_tensor.cpu().numpy())])
                x_array = TaskSettingObj.test_df.drop([TaskSettingObj.target], axis=1).to_numpy()
                gt_y_array = TaskSettingObj.test_df[[TaskSettingObj.target]].to_numpy()
                pred_y_array = clf.predict(x_array)
                print("TABNET MSE = ", np.mean((pred_y_array - gt_y_array) ** 2))
                print("TABNET MAE = ", np.mean(np.abs(pred_y_array - gt_y_array)))
                print("TABNET R2 = ", np.corrcoef(pred_y_array.T, gt_y_array.T)[0, 1] ** 2)
                return
            # saved_filepath = clf.save_model("./tabnet.pth")
            model = MultiBranchModel(
                input_dim_lst=[len(_) for _ in col_group_lst],
                first_dim=50 + step,
                second_dim=20 + step
            ).cuda()
            train = Train(
                model=model,
                criterion=criterion,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                l1_alpha=l1_alpha,
                batch_size=batch_size
            )
            train.train(epoch=epoch_num, lr=1e-4 * lr, is_add_l1=is_add_l1)
            model.cpu()
            with open(prefix + str(0) + ".pth", "wb+") as f:
                torch.save(model, f)


if __name__ == "__main__":
    train_baseline_obj = TrainBaselineObj()
    train_baseline_obj.start_train()
    train_baseline_obj.save_ensemble_model()
    TrainBaselineObj.test_baseline_model()
