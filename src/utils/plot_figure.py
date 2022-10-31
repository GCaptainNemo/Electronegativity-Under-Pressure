import matplotlib.pyplot as plt
import numpy as np


def plt_result(predict_data, gt_data, title="Ground Truth - Predict figure", ):
    plt.scatter(predict_data, gt_data,  marker="o",)
    plt.xlabel("Predict")
    plt.ylabel("Ground Truth")
    plt.title(title)
    min_val = min(np.min(predict_data), np.min(gt_data))
    max_val = max(np.max(predict_data), np.max(gt_data))
    plt.plot((min_val, max_val), (min_val, max_val), c="r")
    plt.show()


if __name__ == "__main__":
    a = np.random.random([10, 1])
    b = np.random.random([10, 1])
    plt_result(a, b)
