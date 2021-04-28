# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：3/31/2021 8:45 PM
# Tool ：PyCharm
import torch
import numpy as np

def egs():
	t1 = torch.Tensor([[1, 2, 3], [1, 2, 3]])
	print(t1)
	"""array1 = np.array(12)
	t_array = torch.Tensor(array1)
	print(t_array)"""

	t2 = torch.empty(5, 8)
	t3 = torch.rand([4, 8])
	t4 = torch.randint(low=-10, high=10, size=[5, 3])
	t5 = torch.randn([3, 4])
	print(t2, t3, t4, t5)


if __name__ == '__main__':
    egs()