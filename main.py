import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import glob
import os
from PIL import Image
import numpy as np


batch_size = 100


# Нейронная сеть с её архитектурой
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 10)

    def forward(self, x):
        # x=torch.FloatTensor(x)
        x=x.view(-1,100)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

    def name(self):
        return "MLP"


#определяем значение метки под каждый класс
label_mark = {
    0: [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
    1: [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]],
    2: [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]],
    3: [[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]],
    4: [[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]],
    5: [[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]],
    6: [[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]],
    7: [[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]],
    8: [[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
    9: [[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]],
}

X_train = []
Y_train = []


images_dir = 'data_'
image_dim = 10
images = []

jpg_filepaths = glob.glob(os.path.join(images_dir, '*.jpg'))
anotaions = glob.glob(os.path.join(images_dir, '*.txt'))

anotaions=sorted(anotaions)
jpg_filepaths=sorted(jpg_filepaths)



X = []

#Считываем файл с данными
for filepath in jpg_filepaths:
	image = Image.open(filepath).resize((image_dim, image_dim));
	# нормализация картинки
	images.append( (np.asarray(image)) );

    
# Считываем анотационный файл
for filepath in anotaions:
    myfile = open(filepath, "rb")
    for line in myfile:
        mini_label=int(line)
        Y_train.append(label_mark[mini_label])


images_dir = 'test_'
image_dim = 10


test_jpg_filepaths = glob.glob(os.path.join(images_dir, '*.jpg'))
test_anotaions = glob.glob(os.path.join(images_dir, '*.txt'))

X_test=[]
Y_test=[]

#Считываем файл с данными
for filepath in test_jpg_filepaths:
	image = Image.open(filepath).resize((image_dim, image_dim));
	# нормализация картинки
	X_test.append( (np.asarray(image)) );

    # Считываем анотационный файл
for filepath in test_anotaions:
    myfile = open(filepath, "rb")
    for line in myfile:
        mini_label=int(line)
        Y_test.append(label_mark[mini_label])


X_Train=images
Y_Train=np.asarray(Y_train)
Y_test=np.asarray(Y_test)
trainloader = zip(X_Train, Y_Train)
test_loader = zip(X_test,Y_test)

net = MLPNet()

#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

#создаем нейронную сеть
net.train()
loss_sum = 0
acc_sum = 0

#Подбираем метод оптимизации
optimizer  = optim.Adamax(net.parameters())#Метод оптимизации
criterion = nn.MSELoss()# Mean Squared Error - критерий оптимальности


running_loss = 0.0
net.train()


loss_history = []
acc_history = []

def train(epoch):
    net.train() 
    
    for batch_id, (data, label) in enumerate(trainloader, 0):
        data = Variable(torch.from_numpy(data.astype(np.float32)))
        target = Variable(torch.from_numpy(label.astype(np.float32)))
        
        
        optimizer.zero_grad()
        preds = net(data)
        #определяем ошибку между Истинным значением и Результатов выдачи нейронной сети
        loss = criterion(preds, target)
        #обратное распространение ошибки
        loss.backward()
        loss_history.append(loss.data)
        #увеличиваем шаг
        optimizer.step()
        
        if batch_id % 100 == 0:
            print(loss.data)

#определяем класс победитель
def pred___(l):
  m = max(l[0])
  return l[0].index(m)

def test(epoch):
    net.eval() 
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:

        data = Variable(torch.from_numpy(data.astype(np.float32)), volatile=True) 
        target = Variable(torch.from_numpy(target.astype(np.float32)))
        
        output = net(data)
        test_loss += criterion(output, target).data

        pred__=pred___(output.tolist())
        real=pred___(target.tolist())
        if (pred__==real):
            correct =correct+1

        #Дополним функционал
        print("Предсказанное число:",pred__)
        print("Реальное число:",real)


    test_loss = test_loss
    test_loss /= len(X_test) 
    accuracy = 100. * correct / len(X_test)
    acc_history.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(X_test),
        accuracy))


for epoch in range(0, 1):
    print("Epoch %d" % epoch)
    train(epoch)
    test(epoch)



