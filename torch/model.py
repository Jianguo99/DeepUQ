import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from torch.utils import data
import joblib
import os
import random
class FNN(nn.Module):
    def __init__(self,D,L,d) -> None:
        super().__init__()
        self.D = int(D) 
        self.L = int(L) 
        self.d = int(d) 
        self.rho = (1./self.L)*np.log(self.d/(self.D*1.))
        # 定义网络节点数
        self.ds = [self.D] + [int(np.ceil(self.D*np.exp(self.rho*i))) for i in range(1, self.L+1)] +\
                  [300*self.d, 1]  # link function
        self.seq_model = nn.Sequential()
        for i in range(1, len(self.ds)):
            self.seq_model.add_module(str(i),nn.Linear(self.ds[i-1],self.ds[i]))
            if i !=  len(self.ds)-3 and i!= len(self.ds)-1:   # 倒数第二层网络
                self.seq_model.add_module(str(i)+"_Relu",nn.ReLU())

    def forward(self, x):
        
        output = self.seq_model(x)
        return output





class LossFNN(nn.Module):
    def __init__(self,D,L,d) -> None:
        super().__init__()
        self.D = int(D) 
        self.L = int(L) 
        self.d = int(d) 
        self.rho = (1./self.L)*np.log(self.d/(self.D*1.))
        # 定义网络节点数
        self.ds = [self.D] + [int(np.ceil(self.D*np.exp(self.rho*i))) for i in range(1, self.L+1)]
        self.seq_model = nn.Sequential()
        for i in range(1, len(self.ds)):
            self.seq_model.add_module(str(i),nn.Linear(self.ds[i-1],self.ds[i]))
            if i!= len(self.ds)-1:   # 最后一层不需要激活函数
                self.seq_model.add_module(str(i)+"_Relu",nn.ReLU())

    def forward(self, x):
        
        output = self.seq_model(x)
        return output


class LossFNNSwish(nn.Module):
    def __init__(self,D,L,d) -> None:
        super().__init__()
        self.D = int(D) 
        self.L = int(L) 
        self.d = int(d) 
        self.rho = (1./self.L)*np.log(self.d/(self.D*1.))
        # 定义网络节点数
        self.ds = [self.D] + [int(np.ceil(self.D*np.exp(self.rho*i))) for i in range(1, self.L+1)] +[1]
        self.seq_model = nn.Sequential()
        for i in range(1, len(self.ds)):
            self.seq_model.add_module(str(i),nn.Linear(self.ds[i-1],self.ds[i]))
            if i!= len(self.ds)-1:   # 最后一层不需要激活函数
                self.seq_model.add_module(str(i)+"_SILU",nn.SiLU())

    def forward(self, x):
        
        output = self.seq_model(x)
        return output
    
class MyDataset(data.Dataset):
    def __init__(self,datadir,TrainDataFile) -> None:
        super().__init__()
        self.TrainDataFile = TrainDataFile
        if isinstance(self.TrainDataFile,list):
            TrainData1 = joblib.load(os.path.join(datadir,TrainDataFile[0]))
            TrainData2 = joblib.load(os.path.join(datadir,TrainDataFile[1]))
            self.TrainData3 = joblib.load(os.path.join(datadir,TrainDataFile[2]))

            self.Inputs=TrainData1["Inputs"]
            self.labels=TrainData2["labels"]
        else:
            TrainData = joblib.load(os.path.join(datadir,self.TrainDataFile))

            self.Inputs=TrainData["Inputs"]
            self.labels=TrainData["labels"]
        self.transform  = transforms.Compose([
            transforms.ToTensor(),      # 这里仅以最基本的为例
        ])

    def __getitem__(self,index):
        """返回一个样本的数据"""
        if isinstance(self.TrainDataFile,list):
            lengthscale_index= int(index//(1024*100))
            # print(np.array(self.Inputs[index]))
            # print(self.TrainData3[lengthscale_index])
            return np.concatenate((np.array(self.Inputs[index]),self.TrainData3[lengthscale_index]),axis=0), self.labels[index].reshape(1)
        else:
            return self.Inputs[index], self.labels[index].reshape(1)

    def __len__(self,):
        return len(self.labels)


def weigth_init(m):
    """
    初始化模型canshu
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
