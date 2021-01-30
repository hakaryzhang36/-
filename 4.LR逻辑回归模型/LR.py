
# coding: utf-8

# # Logistic regression 逻辑回归模型 

# In[31]:


import torch


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## 1.读取数据

# In[33]:


data = pd.read_csv('credit-a.csv', header=None)


# In[34]:


data


# In[35]:


data.info()


# In[36]:


X = data.iloc[:, :-1]  # 取出特征值，iloc[第x1行:第x2行, 第y1列:第y2列]


# In[37]:


Y = data.iloc[:, -1]  # 取出目标值


# In[38]:


Y = Y.replace(-1, 0)  # 把-1替换为0


# ## 2.数据预处理

# In[39]:


X = torch.from_numpy(X.values.astype(np.float32))


# In[40]:


Y = torch.from_numpy(Y.values.reshape(-1, 1).astype(np.float32))


# In[41]:


X.shape


# In[42]:


Y.shape


# ## 3.初始化模型、损失函数、目标函数

# In[44]:


from torch import nn


# In[45]:


model = nn.Sequential(
    nn.Linear(15, 1),  # 线性层，输入特征数15，输出数1
    nn.Sigmoid() # Sigmoid激活层
)


# <img src="https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=1883846040,893574025&fm=15&gp=0.jpg", width="50%">
# <img src="https://5b0988e595225.cdn.sohucs.com/images/20181019/a4fe1ff6079142908d7ec4b97fbfb01c.jpeg", width="30%">
# $$Sigmoid函数$$

# In[47]:


loss_fn = nn.BCELoss()  # 二元交叉熵损失函数


# In[48]:


opt = torch.optim.Adam(model.parameters(), lr=0.0001)


# ## 4.训练模型

# In[53]:


batches = 16
num_of_batch = 653//16


# In[54]:


epoches = 1000


# In[55]:


for epoches in range(epoches):
    for batch in range(num_of_batch):
        start = batches*batch  # 起始序号
        end = start + batches  # 结束序号
        x = X[start: end]
        y = Y[start: end]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


# In[56]:


# 模型状态
model.state_dict()


# ## 5.检查结果

# In[60]:


# 正确率
((model(X).data.numpy() > 0.5).astype("int") == Y.numpy()).mean()


# In[ ]:




