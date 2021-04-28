# coding: utf-8
# Team : Quality Management Center
# Author：Guo Zikun
# Email: gzk798412226@gmail.com
# Date ：2021/4/26 16:55
# Tool ：PyCharm
import torch
from torch import nn
from torch.nn import functional as F
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        #b 16, 5, 5
        print(out.shape)
        """criteon, 评价标准"""
        self.criteon = nn.CrossEntropyLoss()


    def forward(self, x):
        """

        :param x:[b, 3,32,32]
        :return:
        """
        batchsz = x. size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz, 16*5*5)
        logits = self.fc_unit(x)

        """ pred = F.softmax(logits, dim=1)"""
        """loss = self.criteon(logits, y)"""
        return logits

if __name__ == '__main__':
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print(out.shape)