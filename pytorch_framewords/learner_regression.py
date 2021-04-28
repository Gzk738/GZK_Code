# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：4/1/2021 4:51 PM
# Tool ：PyCharm
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import SGD



"""定义模型"""
class Mymodel(nn.Module):
	def __init__(self):
		super(Mymodel, self).__init__()
		self.linear = nn.Linear(1, 1)
	def forward(self, x):
		out = self.linear(x)
		return out

def handle_learnerregression():
	learn_rate = 0.01
	"""准备数据"""
	x =  torch.rand([500, 1])
	y_true = x*3 + 0.8
	"""模型计算"""
	w = torch.rand([1, 1], requires_grad=True)
	b = torch.tensor(0, requires_grad=True, dtype=torch.float32)

	"""训练"""
	for i in range(200):
		y_predict = torch.matmul(x, w) + b
		loss = (y_true - y_predict).pow(2).mean()
		if w.grad is not None:
			w.grad.data.zero_()
		if b.grad is not None:
			b.grad.data.zero_()

		loss.backward()
		w.data = w.data - learn_rate*w.grad
		b.data = b.data - learn_rate*b.grad

		if i % 50 == 0:
			print(w.item(), b.item(), loss.item())

	plt.figure(figsize = (20, 8))
	plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1))
	y_predict = torch.matmul(x, w) + b
	plt.plot(x.numpy().reshape(-1), y_predict.detach().numpy().reshape(-1), c = "red")
	plt.show()
	return 0

def api_learner_regression():
	"""
	api实现线性回归
	:return:
	"""
	"""数据准备"""
	x = torch.rand([500, 1])
	y_true = 3*x + 0.8

	"""实例化模型"""
	my_model = Mymodel()
	optimizer = SGD(my_model.parameters(), 0.01)
	loss_fn = nn.MSELoss()

	for i in range(200):

		y_predict = my_model(x)
		loss = loss_fn(y_predict, y_true)
		"""梯度置为0"""
		optimizer.zero_grad()
		"""反向传播"""
		loss.backward()
		"""参数更新"""
		optimizer.step()
		if i%10 == 0:
			params = list(my_model.parameters())
			"""parameters就是参数"""
			print(loss.item, params[0].item(), params[1].item())




if __name__ == '__main__':
	#api_learner_regression()
	handle_learnerregression()












