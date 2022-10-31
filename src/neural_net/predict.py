from src.neural_net.data import CustomDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from src.settings.settings import TaskSettingObj


class KFoldBaggingModel(nn.Module):
	def __init__(self, *model_lst):
		"""
		k-fold Bagging ensemble learning technique
		"""
		super(KFoldBaggingModel, self).__init__()
		self.model_lst = nn.ModuleList([model for model in model_lst])

	def forward(self, x):
		res = torch.cat([model(x) for model in self.model_lst], dim=1)
		# print(res.shape)
		return torch.mean(res, dim=1, keepdim=True)


class KFoldBaggingModelTest(nn.Module):
	def __init__(self, *model_lst):
		"""
		use k-fold bagging ensemble k Base Learner
		"""
		super(KFoldBaggingModelTest, self).__init__()
		self.model_lst = nn.ModuleList([model for model in model_lst])

	def forward(self, x):
		res = torch.cat([torch.unsqueeze(model(x), dim=2) for model in self.model_lst], dim=2)
		return torch.mean(res, dim=2, keepdim=False)


class NGroupModel(nn.Module):
	def __init__(self, partition_lst, *model_lst):
		"""
		partition_lst: e.g., [0, 2, 5, 7, 9] => x[:, 0:2] input first model, x[:, 2:5] input second model, ...
		"""
		super(NGroupModel, self).__init__()
		self.model_lst = nn.ModuleList([model for model in model_lst])
		self.partition_lst = partition_lst
		print(self.partition_lst)
		print(self.n_group_model)

		# self.k = len(self.k_fold_model)

	def forward(self, x):
		res = torch.cat([model(x[:, self.partition_lst[i]:self.partition_lst[i + 1]])
						 for i, model in enumerate(self.n_group_model)], dim=1)
		# print(res.shape)
		return torch.mean(res, dim=1, keepdim=True)


def get_loss(
	test_dataframe,
	col_names_lst,
	target,
	model,
	loss_func,
	batch_size=TaskSettingObj.validation_batch_size,
	is_cuda=True,
	):
	"""
	Args:
		test_dataframe： test set
		col_names_lst：input columns
		target: output columns
		model: trained model
		loss_func: loss function
	return:
		loss
	"""
	test_dataset = CustomDataset(test_dataframe, col_names_lst, target, is_cuda=False)
	test_data_loader = DataLoader(
		dataset=test_dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False)
	dataset_num = len(test_dataset)
	avg_loss = 0
	if is_cuda:
		model = model.cuda()
		with torch.no_grad():
			for i, (x, gt_y) in enumerate(test_data_loader):
				x, gt_y = x.cuda(), gt_y.cuda()
				pred_y = model(x)
				loss = loss_func(pred_y, gt_y) * x.shape[0] / dataset_num
				avg_loss += loss.cpu()
		return avg_loss
	else:
		with torch.no_grad():
			for i, (x, gt_y) in enumerate(test_data_loader):
				pred_y = model(x)
				loss = loss_func(pred_y, gt_y) * x.shape[0] / dataset_num
				avg_loss += loss
		return avg_loss


def get_predict(test_dataframe, col_names_lst, target, model,
				batch_size=TaskSettingObj.validation_batch_size,
				is_cuda=True):
	"""
	Args:
		test_dataframe： test set
		col_names_lst：input columns
		target: output columns
		model: trained model
		loss_func: loss function
	return:
		pred_y, gt_y
	"""
	test_dataset = CustomDataset(test_dataframe, col_names_lst, target, is_cuda=False)
	test_data_loader = DataLoader(
		dataset=test_dataset,
		batch_size=batch_size,
		shuffle=False,
		drop_last=False)

	pd_y_lst, gt_y_lst = [], []
	if not is_cuda:
		model = model.cpu()
		with torch.no_grad():
			for i, (x, gt_y) in enumerate(test_data_loader):
				pred_y = model(x)
				# # seebeck符号先验
				# pred_y[torch.where(pred_y * x[:, 1] < 0)] = 0
				pd_y_lst.append(pred_y)
				gt_y_lst.append(gt_y)
	else:
		model = model.cuda()
		with torch.no_grad():
			for i, (x, gt_y) in enumerate(test_data_loader):
				x = x.cuda()
				pred_y = model(x)
				pred_y = pred_y.cpu()
				pd_y_lst.append(pred_y)
				gt_y_lst.append(gt_y)
	return torch.cat(pd_y_lst, dim=0), torch.cat(gt_y_lst, dim=0)


def load_k_fold_bagging_model(prefix, base_learner_num):
	"""
	load k-fold bagging model
	"""
	model_lst = []
	for i in range(base_learner_num):
		model_name = prefix + str(i) + ".pth"
		# model_name = prefix + str(0) + ".pth"
		model = torch.load(model_name)
		model = model.to("cpu")
		model.eval()
		model_lst.append(model)
	k_fold_bagging_model = KFoldBaggingModel(*model_lst)
	return k_fold_bagging_model


def get_n_group_model(column_group_lst, *model_lst):
	"""
	get n group model
	"""
	partition_lst = [0]
	for column_group in column_group_lst:
		partition_lst.append(partition_lst[-1] + len(column_group))
	n_group_model = NGroupModel(partition_lst, *model_lst)
	return n_group_model
