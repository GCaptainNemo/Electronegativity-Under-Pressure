import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0):
        """
        Args:
        patience (int): How long to wait after last time validation loss improved.
                            Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.epoch = 0
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        self.epoch += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience and self.epoch > 700:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print("score = {} < {} + {}".format(score, self.best_score, self.delta))
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss