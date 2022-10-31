import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from src.neural_net.early_stop import EarlyStopping
from src.settings.settings import TaskSettingObj


class Train:
    def __init__(self, model, criterion, train_dataset, val_dataset, batch_size=64, l1_alpha=1e-8):
        self.criterion = criterion
        self.model = model
        self.train_data_loader = DataLoader(dataset=train_dataset,	
                              batch_size=batch_size, shuffle=True, drop_last=False)
        self.val_data_loader = DataLoader(dataset=val_dataset,
                              batch_size=len(val_dataset), shuffle=False, drop_last=False)
        self.l1_alpha = l1_alpha
        # ################################################################################
        # whether add noise in the trainset
        if TaskSettingObj.is_data_aug and TaskSettingObj.is_normalize_x and \
                TaskSettingObj.is_normalize_y:
            self.is_data_aug = True
            self.x_data_aug_std = TaskSettingObj.x_data_aug_noise_std
            self.y_data_aug_std = TaskSettingObj.y_data_aug_noise_std
        else:
            self.is_data_aug = False

    def train(self, epoch, lr, is_add_l1=False):
        early_stopping_obj = EarlyStopping()
        optimizer = optim.Adam(self.model.parameters(), lr)
        if not is_add_l1:
            for e in range(epoch):
                self.model.eval()
                avg_loss = 0
                total_num = 0
                with torch.no_grad():
                    for i, (x, gt_y) in enumerate(self.val_data_loader):
                        x, gt_y = x.cuda(), gt_y.cuda()
                        pred_y = self.model(x)
                        loss = self.criterion(pred_y, gt_y)
                        avg_loss += loss
                        total_num += x.shape[0]
                    avg_loss /= (i + 1)
                    print("Epoch {}  -  loss: {} ".format(e, avg_loss))
                    early_stopping_obj(avg_loss)
                    if early_stopping_obj.early_stop:
                        print("Early stopping!!!")
                        break
                self.model.train()
                # ############################################
                for i, (x, gt_y) in enumerate(self.train_data_loader):
                    x, gt_y = x.cuda(), gt_y.cuda()
                    if self.is_data_aug:
                        x_noise = torch.randn(x.shape[1], device=x.device) * self.x_data_aug_std
                        y_noise = torch.randn(gt_y.shape[1], device=gt_y.device) * self.y_data_aug_std
                        x += x_noise
                        gt_y += y_noise
                    # print(x, x.shape)
                    optimizer.zero_grad()
                    pred_y = self.model(x)

                    loss = self.criterion(pred_y, gt_y)
                    loss.backward()
                    optimizer.step()
        else:
            for e in range(epoch):
                self.model.eval()
                avg_loss = 0
                total_num = 0
                with torch.no_grad():
                    for i, (x, gt_y) in enumerate(self.val_data_loader):
                        x, gt_y = x.cuda(), gt_y.cuda()
                        pred_y = self.model(x)
                        loss = self.criterion(pred_y, gt_y)
                        avg_loss += loss
                        total_num += x.shape[0]
                    avg_loss /= (i + 1)
                    print("Epoch {}  -  loss: {} ".format(e, avg_loss))
                    early_stopping_obj(avg_loss)
                    if early_stopping_obj.early_stop:
                        print("Early stopping!!!")
                        break
                self.model.train()
                # ############################################
                for i, (x, gt_y) in enumerate(self.train_data_loader):
                    x, gt_y = x.cuda(), gt_y.cuda()
                    if self.is_data_aug:
                        x_noise = torch.randn(x.shape[1], device=x.device) * self.x_data_aug_std
                        y_noise = torch.randn(gt_y.shape[1], device=gt_y.device) * self.y_data_aug_std
                        x += x_noise
                        gt_y += y_noise
                    # print(x, x.shape)
                    optimizer.zero_grad()
                    regularization_loss = 0
                    for param in self.model.parameters():
                        regularization_loss += torch.sum(abs(param))
                    pred_y = self.model(x)
                    loss = self.criterion(pred_y, gt_y) + self.l1_alpha * regularization_loss
                    loss.backward()
                    optimizer.step()


