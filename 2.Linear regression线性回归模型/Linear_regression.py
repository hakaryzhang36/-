
# coding: utf-8

# # Linear regression模型

# In[1]:


import torch


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## 1.读取数据

# In[3]:


data = pd.read_csv('Income1.csv')


# ## 2.查看数据信息

# In[4]:


data.info()


# ## 3.查看数据

# In[5]:


data


# ## 4.生成图表

# In[10]:


plt.scatter(data.Education,data.Income)
plt.xlabel('Education')
plt.ylabel('Income')


# ## 5.数据预处理，类型转换
# array(30,)->array(30,1)->tensor(30,0)

# In[11]:


data.Education.values


# In[12]:


data.Education.values.shape


# In[13]:


data.Education.values.reshape(-1,1)


# In[14]:


data.Education.values.reshape(-1,1).shape


# In[15]:


from torch import nn


# In[16]:


X = torch.from_numpy(data.Education.values.reshape(-1,1).astype(np.float32))
X


# In[17]:


# 同样处理Income
Y = torch.from_numpy(data.Income.values.reshape(-1,1).astype(np.float32))


# ## 6.初始化模型、损失函数、目标函数

# In[18]:


model = nn.Linear(1, 1) # 线性模型Y = W * X + b 等价于 model(input)


# In[19]:


loss_fn = nn.MSELoss() # 损失函数，均方误差


# In[20]:


opt = torch.optim.SGD(model.parameters(), lr=0.0001) # 目标函数，指定优化参数，parameters方法返回参数，lr为学习率


# ## 7.训练模型

# In[21]:


for epoch in range(5000):
    for x, y in zip(X, Y):
        y_pred = model(x)           # 使用模型预测
        loss = loss_fn(y, y_pred)   # 根据预测结果计算损失
        opt.zero_grad()             # 梯度清零
        loss.backward()             # 反向传播求解梯度
        opt.step()                  # 优化模型


# 查看结果

# In[22]:


model.weight


# In[23]:


model.bias


# ## 8.图表展示

# In[24]:


plt.scatter(data.Education, data.Income)
plt.plot(X.numpy(), model(X).data.numpy(), c='r')


# In[ ]:




