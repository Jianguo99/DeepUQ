import joblib
from model import FNN, MyDataset,weigth_init,setup_seed
import numpy as np
import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from Regularizer import L1Regularizer,L2Regularizer
# from keras.backend.tensorflow_backend import set_session ## 和tf.keras不同

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# 设置随机数种子
setup_seed(20)
l_N = 60   # lengthscale对的数量
k_ ='rbf'
nx = 32
ny = 32
num_samples = 100   # 每个随机场采样的样本数


TrainDataFile = k_+"_l_N_"+str(l_N)+"_num_samples_"+str(num_samples)+\
                    "_nx_"+str(nx)+"_ny_"+str(ny)+"_train.pkl"

save_model_name = k_+"_l_N_"+str(l_N)+"_num_samples_"+str(num_samples)+\
                    "_nx_"+str(nx)+"_ny_"+str(ny)+"_train_meta.pth"
datadir = "/home/huangjg/MyFiles/deep-uq-paper/data"

DataSet = MyDataset(datadir,TrainDataFile)

####################################
#######training model##############
################################

BATCH_SIZE=1024
EPOCHS = 300  # 训练迭代次数
# dataset = tf.data.Dataset.from_tensor_slices((Inputs,label))
# batched_dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
Input_Dim = nx*ny+2
TrainLoader=  torch.utils.data.DataLoader(DataSet,batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=False)


MyModel = FNN(1026,7,2)
meta_model_path = k_+"_l_N_"+str(l_N)+"_num_samples_"+str(num_samples)+\
                    "_nx_"+str(nx)+"_ny_"+str(ny)+"_train_meta.pth"
MyModel.load_state_dict(torch.load(os.path.join("/home/huangjg/MyFiles/deep-uq-paper/meta_torch/MyReptile","pth",meta_model_path)))  #  保存模型参数)

MyModel.cuda()

print("模型初始化成功！ 数据加载成功！")
# print(MyModel)
LR = 0.001
optimizer =  torch.optim.Adam(MyModel.parameters(),LR,
                                            betas = (0.5,0.9),
                                            eps=1e-08,
                                            weight_decay=0)
criterion = F.mse_loss  

l1_regularizer = L1Regularizer(MyModel,lambda_reg=1e-6)
l2_regularizer = L2Regularizer(MyModel,lambda_reg=1e-6)
MyModel.train()
loss_best = np.inf
loss_list= []
print(len(TrainLoader))
for epoch in range(EPOCHS):
    with tqdm(total=len(TrainLoader)) as _tqdm: # 使用需要的参数对tqdm进行初始化
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, EPOCHS))# 设置前缀 一般为epoch的信息
        ### 训练过程#####
        avg_loss =0
        for j ,(input,label) in enumerate(TrainLoader):
            input =  torch.from_numpy(np.array(input,dtype=np.float32)).cuda()
            label =  torch.from_numpy(np.array(label,dtype=np.float32)).cuda()
            optimizer.zero_grad()  # 将模型的参数的梯度初始化为0
            pred = MyModel(input)   #输出预测结果 

            loss = criterion(pred,label)   # 计算loss
            l1_regularizer.regularized_all_param(loss)
            l2_regularizer.regularized_all_param(loss)
            loss.backward()   # 反向传播

            optimizer.step()   #优化
            avg_loss = (avg_loss*np.maximum(0,j) + loss.data.cpu().numpy())/(j+1)
            _tqdm.set_postfix(loss='{:.6f}'.format(avg_loss)) # 设置你想要在本次循环内实时监视的变量  可以作为后缀打印出来
            _tqdm.update(1)  # 设置你每一次想让进度条更新的iteration 大小
        loss_list.append(avg_loss)
        if avg_loss < loss_best and epoch != 0:
            torch.save(MyModel.state_dict(),os.path.join("/home/huangjg/MyFiles/deep-uq-paper/torch","pth",save_model_name) ) #  保存模型参数
        loss_best =  avg_loss
        print("保存第%d个周期的模型参数!"%epoch)

joblib.dump(loss_list,os.path.join("/home/huangjg/MyFiles/deep-uq-paper/torch/pth","loss_list_meta.pkl"))

# for epoch in range(EPOCHS):
#         ### 训练过程#####
#     avg_loss =0
#     for j ,(input,label) in enumerate(TrainLoader):
#         optimizer.zero_grad()  # 将模型的参数的梯度初始化为0
#         pred = MyModel(input)   #输出预测结果 

#         loss = criterion(pred,label)   # 计算loss
#         l1_regularizer.regularized_all_param(loss)
#         l2_regularizer.regularized_all_param(loss)
#         loss.backward()   # 反向传播

#         optimizer.step()   #优化
#         avg_loss = (avg_loss*np.maximum(0,j) + loss.data.cpu().numpy())/(j+1)
#     loss_list.append(avg_loss)
#     if avg_loss < loss_best and epoch != 0:
#         torch.save(MyModel.state_dict(),os.path.join("/home/huangjg/MyFiles/deep-uq-paper/torch","pth",os.path.splitext(TrainDataFile)[0]+".pth"))  #  保存模型参数
#     loss_best =  avg_loss
#     print("保存第%d个周期的模型参数!"%epoch)