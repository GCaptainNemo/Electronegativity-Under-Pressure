from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, pd, column_names, target, is_cuda=True):
        super(CustomDataset, self).__init__()
        # #################################
        # header
        # #################################
        self.column_names = column_names
        self.target = target
        self.X_tensor = torch.tensor(np.array(pd[self.column_names]), dtype=torch.float32)
        self.Y_tensor = torch.tensor(np.array(pd[[self.target]]), dtype=torch.float32)
        if is_cuda:
            self.X_tensor = self.X_tensor.cuda()
            self.Y_tensor = self.Y_tensor.cuda()

    def __len__(self):
        return self.X_tensor.shape[0]

    def __getitem__(self, item):
        return self.X_tensor[item, :], self.Y_tensor[item, :]


if __name__ == "__main__":
    # data_sampler = CustomDataset(train_df)
    # data_sampler = DataLoader(dataset=data_sampler,
    #                           batch_size=4, shuffle=True, drop_last=True)
    #
    # for x, y in data_sampler:
    #     print("x = ", x, x.shape)
    #     print("y = ", y, y.shape)
    ...



