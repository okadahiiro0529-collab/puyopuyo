import torch
import torch.nn as nn
import torch.nn.functional as F


class C4Net_6x7(nn.Module):

	def __init__(self, num_inputs=None, num_actions=None):
		super(C4Net_6x7, self).__init__()

		self.main = nn.Sequential()

		self.main.add_module('Conv_1', nn.Conv2d(1, 64, 3, stride=1, padding=1))
		self.main.add_module('bn_1', nn.BatchNorm2d(64))
		self.main.add_module('relu_1', nn.ReLU())

		self.main.add_module('Conv_2', nn.Conv2d(64, 128, 3, stride=1, padding=1))
		self.main.add_module('bn_2', nn.BatchNorm2d(128))
		self.main.add_module('relu_2', nn.ReLU())

		self.main.add_module('Conv_3', nn.Conv2d(128, 256, 3, stride=1, padding=1))
		self.main.add_module('bn_3', nn.BatchNorm2d(256))
		self.main.add_module('relu_3', nn.ReLU())

		self.main.add_module('Conv_4', nn.Conv2d(256, 512, 3, stride=1))
		self.main.add_module('bn_4', nn.BatchNorm2d(512))
		self.main.add_module('relu_4', nn.ReLU())

		self.main.add_module('Conv_5', nn.Conv2d(512, 512, (2, 3), stride=1))
		self.main.add_module('bn_5', nn.BatchNorm2d(512))
		self.main.add_module('relu_5', nn.ReLU())

		self.main.add_module('Conv_6', nn.Conv2d(512, 512, 3, stride=1))
		self.main.add_module('relu_6', nn.ReLU())

		self.fc1 = nn.Linear(512, 256)
		self.policy_output = nn.Linear(256, num_actions)
		self.value_output = nn.Linear(256, 1)


	def forward(self, inputs):

		x = inputs
		x = self.main(x)
		
		x = x.view(-1, 512)
		x = F.dropout(F.relu(self.fc1(x)), p=0.3, training=self.training)

		return F.softmax(self.policy_output(x), dim=1), F.tanh(self.value_output(x))

class C4Net_5x5(nn.Module):

	def __init__(self, num_inputs=None, num_actions=None):
		super(C4Net_5x5, self).__init__()

		self.main = nn.Sequential()

		self.main.add_module('Conv_1', nn.Conv2d(1, 64, 3, stride=1, padding=1))
		self.main.add_module('bn_1', nn.BatchNorm2d(64))
		self.main.add_module('relu_1', nn.ReLU())

		self.main.add_module('Conv_2', nn.Conv2d(64, 128, 3, stride=1, padding=1))
		self.main.add_module('bn_2', nn.BatchNorm2d(128))
		self.main.add_module('relu_2', nn.ReLU())

		self.main.add_module('Conv_3', nn.Conv2d(128, 256, 3, stride=1, padding=1))
		self.main.add_module('bn_3', nn.BatchNorm2d(256))
		self.main.add_module('relu_3', nn.ReLU())

		self.main.add_module('Conv_4', nn.Conv2d(256, 512, 3, stride=1))
		self.main.add_module('bn_4', nn.BatchNorm2d(512))
		self.main.add_module('relu_4', nn.ReLU())

		self.main.add_module('Conv_5', nn.Conv2d(512, 512, 3, stride=1))
		self.main.add_module('relu_5', nn.ReLU())

		self.fc1 = nn.Linear(512, 256)
		self.policy_output = nn.Linear(256, num_actions)
		self.value_output = nn.Linear(256, 1)


	def forward(self, inputs):

		x = inputs
		x = self.main(x)
		
		x = x.view(-1, 512)
		x = F.dropout(F.relu(self.fc1(x)), p=0.3, training=self.training)

		return F.softmax(self.policy_output(x), dim=1), F.tanh(self.value_output(x))


class XandosNet(nn.Module):

	def __init__(self, num_inputs=None, num_actions=9):
		super(XandosNet, self).__init__()

		self.main = nn.Sequential()

		self.main.add_module('Conv_1', nn.Conv2d(1, 64, 3, stride=1, padding=1))
		self.main.add_module('bn_1', nn.BatchNorm2d(64))
		self.main.add_module('relu_1', nn.ReLU())

		self.main.add_module('Conv_2', nn.Conv2d(64, 128, 3, stride=1, padding=1))
		self.main.add_module('bn_2', nn.BatchNorm2d(128))
		self.main.add_module('relu_2', nn.ReLU())

		self.main.add_module('Conv_3', nn.Conv2d(128, 256, 3, stride=1, padding=1))
		self.main.add_module('bn_3', nn.BatchNorm2d(256))
		self.main.add_module('relu_3', nn.ReLU())

		self.main.add_module('Conv_4', nn.Conv2d(256, 512, 3, stride=1))
		self.main.add_module('relu_4', nn.ReLU())

		self.fc1 = nn.Linear(512, 256)
		self.policy_output = nn.Linear(256, num_actions)
		self.value_output = nn.Linear(256, 1)


	def forward(self, inputs):

		x = inputs
		x = self.main(x)
		
		x = x.view(-1, 512)
		x = F.dropout(F.relu(self.fc1(x)), p=0.3, training=self.training)

		return F.softmax(self.policy_output(x), dim=1), F.tanh(self.value_output(x))
