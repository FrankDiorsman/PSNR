import torch
import torch.nn as nn
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim,output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

x_values = [i for i in range(11)]
x_train = np.array(x_values,dtype = np.float32)
x_train = x_train.reshape(-1,1)
x_train.shape

y_values = [2*i+1 for i in x_values]
y_train = np.array(y_values,dtype = np.float32)
y_train = x_train.reshape(-1,1)
y_train.shape

epochs = 1000
for epoch in range(epochs):
    epoch +=1
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs,labels)

    loss.backward()
    optimizer.step()

    if epoch %50 ==0:
        print('epoch{},loss{}'.format(epoch,loss.item))