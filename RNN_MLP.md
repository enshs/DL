```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
```


```python
from plotly.subplots import make_subplots
import plotly.express as px
```


```python
T = 1000 # Generate a total of 1000 points
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
df=pd.DataFrame([time, x]).T
df.columns=["time", "value"]
fig=px.line(df, x="time", y="value", width=400, height=300)
fig.show()
```





이전 4개 데이터를 적용, 이전값의 수를 tau로 선언 


```python
tau=4
features=torch.zeros((T-tau, tau))
for i in range(tau):
    features[:,i]=x[i:(T-tau+i)]
labels=x[tau:].reshape(-1,1)
features.shape, labels.shape
```




    (torch.Size([996, 4]), torch.Size([996, 1]))




```python
batch_size, n_train = 16, 600
# Only the first `n_train` examples are used for training
xtr, ytr=features[:n_train, :], labels[:n_train]
xte, yte=features[n_train:, :], labels[n_train:]
trDs = TensorDataset(xtr, ytr)
trDl = DataLoader(trDs, batch_size=batch_size)
```


```python
# 가중치 초기화 방법으로 xavier 적용
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
#단순한 MLP(Multiple percetron)모델 설정 
def network():
    net=nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10,1))
    net.apply(init_weights) #init_weights 함수를 Linear 모형에 적용 
    return net
#Loss Function
loss=nn.MSELoss()

def train(net, trDl, loss, epochN, lr):
    opt=optim.Adam(net.parameters(), lr)
    for epoch in range(epochN):
        for X, y in trDl:
            opt.zero_grad()
            l=loss(net(X), y)
            l.backward()
            opt.step()
        if epoch % 10 ==0:
            print(f'epoch:{epoch}, loss:{l}')
        
net=network()
train(net, trDl, loss, 100, 0.01)        
```

    epoch:0, loss:0.09900861978530884
    epoch:10, loss:0.10398634523153305
    epoch:20, loss:0.1018042266368866
    epoch:30, loss:0.10103197395801544
    epoch:40, loss:0.10078120976686478
    epoch:50, loss:0.09983043372631073
    epoch:60, loss:0.09882188588380814
    epoch:70, loss:0.09954214096069336
    epoch:80, loss:0.10029686987400055
    epoch:90, loss:0.10090150684118271



```python
loss(net(xte), yte)    
```




    tensor(0.0719, grad_fn=<MseLossBackward>)




```python
pred=net(xte)
```


```python
df1=torch.cat([pred.detach(), yte], dim=1)
plt.figure(figsize=(10, 6))
plt.plot(range(len(df1)), df1[:,0], label="pred")
plt.plot(range(len(df1)), df1[:,1], label="true")
plt.legend(loc='best')
plt.show()
```


    
![png](output_9_0.png)
    


## k-step-ahead prediction  

위 결과와 같이 바로 직전값들을 사용한 예측의 결과는 관찰치와 유사한 경향을 보입니다 그러나 feature에 관찰치 대신 예측치를 사용하는 경우 예상치 못한 결과를 나타냅니다. 

다음은 이전의 4개 값으로 에측한 것으로 처음에서 600개 이상에서는 예측치를 사용한 것입니다. 


즉, $x_{604}$의 k-step-ahead prediction은  $\hat{x}_{604+k}$입니다. 


```python
multStepPre=torch.zeros(T)
multStepPre[:(n_train+tau)]=x[:(n_train+tau)]
for i in range(n_train+tau, T):
    multStepPre[i]=net(multStepPre[(i-tau):i].reshape(1,-1))
multStepPre[:10]
```




    tensor([ 0.1275, -0.0058, -0.1453,  0.1692, -0.0331,  0.0882,  0.1643,  0.1951,
            -0.0801, -0.1423], grad_fn=<SliceBackward>)




```python
multStepPre.shape
```




    torch.Size([1000])




```python
pre1=net(features)
plt.figure(figsize=(10, 7))
plt.plot(time[4:], labels, label="true")
plt.plot(time[4:], pre1.detach(), color="red", label="1-step")
plt.plot(time, multStepPre.detach(), label="4-step")
plt.xlabel("time")
plt.ylabel("prediction")
plt.legend(loc='best')
plt.show()
```


    
![png](output_13_0.png)
    


위의 예에서 볼 수 있듯이 이것은 굉장한 실패입니다. 예측은 몇 가지 예측 단계 후에 매우 빠르게 상수로 감소합니다. 알고리즘이 제대로 작동하지 않는 이유는 무엇입니까? 이것은 궁극적으로 오류가 쌓이기 때문입니다. 1단계 이후에 약간의 오류 ε1 = ε̄가 있다고 가정해 보겠습니다. 이제 2단계의 입력은 ε1에 의해 교란되므로 일부 상수 c에 대해 $\epsion_2=\bar{\epsilon}+c \epsilon_1$ 순서로 오류가 발생합니다. 오차는 실제 관찰에서 다소 빠르게 발산할 수 있습니다. 이것은 일반적인 현상입니다. 예를 들어, 다음 24시간 동안의 일기 예보는 꽤 정확한 경향이 있지만 그 이상은 정확도가 급격히 떨어집니다. 

## time-series MLP 주가예측에 적용 
위 과정을 주가(삼성전자)의 종가예측을 위해 적용합니다. 


```python
import FinanceDataReader as fdr
```


```python
st=pd.Timestamp(2010,1, 1)
et=pd.Timestamp(2021, 11, 11)
da=fdr.DataReader('005930', st, et)
da.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-04</th>
      <td>16060</td>
      <td>16180</td>
      <td>16000</td>
      <td>16180</td>
      <td>239271</td>
      <td>0.012516</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>16520</td>
      <td>16580</td>
      <td>16300</td>
      <td>16440</td>
      <td>559219</td>
      <td>0.016069</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>16580</td>
      <td>16820</td>
      <td>16520</td>
      <td>16820</td>
      <td>459755</td>
      <td>0.023114</td>
    </tr>
  </tbody>
</table>
</div>




```python
def generateTimesLag(data, nLag):
    df=data
    dfN=pd.DataFrame(df.copy())
    for n in range(1, nLag+1):
        dfN[f'lag{n}']=df.shift(n)
    dfN=dfN[nLag:]
    return(dfN)
```


```python
dat=generateTimesLag(da.Close, 10)
dat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>lag1</th>
      <th>lag2</th>
      <th>lag3</th>
      <th>lag4</th>
      <th>lag5</th>
      <th>lag6</th>
      <th>lag7</th>
      <th>lag8</th>
      <th>lag9</th>
      <th>lag10</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-18</th>
      <td>16860</td>
      <td>16840.0</td>
      <td>16540.0</td>
      <td>15940.0</td>
      <td>16180.0</td>
      <td>15940.0</td>
      <td>16420.0</td>
      <td>16260.0</td>
      <td>16820.0</td>
      <td>16440.0</td>
      <td>16180.0</td>
    </tr>
    <tr>
      <th>2010-01-19</th>
      <td>16460</td>
      <td>16860.0</td>
      <td>16840.0</td>
      <td>16540.0</td>
      <td>15940.0</td>
      <td>16180.0</td>
      <td>15940.0</td>
      <td>16420.0</td>
      <td>16260.0</td>
      <td>16820.0</td>
      <td>16440.0</td>
    </tr>
    <tr>
      <th>2010-01-20</th>
      <td>16680</td>
      <td>16460.0</td>
      <td>16860.0</td>
      <td>16840.0</td>
      <td>16540.0</td>
      <td>15940.0</td>
      <td>16180.0</td>
      <td>15940.0</td>
      <td>16420.0</td>
      <td>16260.0</td>
      <td>16820.0</td>
    </tr>
    <tr>
      <th>2010-01-21</th>
      <td>17000</td>
      <td>16680.0</td>
      <td>16460.0</td>
      <td>16860.0</td>
      <td>16840.0</td>
      <td>16540.0</td>
      <td>15940.0</td>
      <td>16180.0</td>
      <td>15940.0</td>
      <td>16420.0</td>
      <td>16260.0</td>
    </tr>
    <tr>
      <th>2010-01-22</th>
      <td>16500</td>
      <td>17000.0</td>
      <td>16680.0</td>
      <td>16460.0</td>
      <td>16860.0</td>
      <td>16840.0</td>
      <td>16540.0</td>
      <td>15940.0</td>
      <td>16180.0</td>
      <td>15940.0</td>
      <td>16420.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```


```python
def generateTensorData4anal(ind, de,target, testSize=0.2, shuffle=False, randState=None, scaler=MinMaxScaler()):
    x=scaler.fit(ind).transform(ind)
    tar=scaler.fit(ind).transform(target)
    de1=np.array(de).reshape(-1,1)
    y=scaler.fit(de1).transform(de1)
    xtr, xte, ytr, yte=train_test_split(x, y, test_size=0.2, shuffle=False, random_state=randState)
    xtr1, xte1, ytr1, yte1,tar1=map(torch.FloatTensor, [xtr, xte, ytr, yte, tar])
    return([xtr1, xte1, ytr1, yte1, tar1])   
```


```python
xtr1, xte1, ytr1, yte1, tar1=generateTensorData4anal(xPd, yPd, target)
xtr1.shape, xte1.shape, tar1.shape
```




    (torch.Size([2333, 10]), torch.Size([584, 10]), torch.Size([1, 10]))




```python
batSize=200
trDs = TensorDataset(xtr1, ytr1)
trDl = DataLoader(trDs, batch_size=batSize)
```


```python
# 가중치 초기화 방법으로 xavier 적용
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
#단순한 MLP(Multiple percetron)모델 설정 
def network(indColN):
    net=nn.Sequential(nn.Linear(indColN, 64), nn.ReLU(), 
                      nn.Linear(64, 32), nn.ReLU(),
                      nn.Linear(32,1))
    net.apply(init_weights) #init_weights 함수를 Linear 모형에 적용 
    return net
#Loss Function
loss=nn.MSELoss()

def train(net, trDl, loss, epochN, lr):
    opt=optim.Adam(net.parameters(), lr)
    for epoch in range(epochN+1):
        for X, y in trDl:
            opt.zero_grad()
            l=loss(net(X), y)
            l.backward()
            opt.step()
        if epoch % 1000 ==0:
            print(f'epoch:{epoch}, loss:{l}')
        
net1=network(xtr1.shape[1])
train(net1, trDl, loss, 10000, 0.001)        
```

    epoch:0, loss:0.015924453735351562
    epoch:1000, loss:0.00012451941438484937
    epoch:2000, loss:0.00014505768194794655
    epoch:3000, loss:0.0001244048326043412
    epoch:4000, loss:0.00010680943523766473
    epoch:5000, loss:9.823913569562137e-05
    epoch:6000, loss:9.788988973014057e-05
    epoch:7000, loss:8.517003880115226e-05
    epoch:8000, loss:7.764957990730181e-05
    epoch:9000, loss:8.11457575764507e-05
    epoch:10000, loss:8.685026841703802e-05



```python
loss(net1(xte1), yte1)
```




    tensor(0.0003, grad_fn=<MseLossBackward>)




```python
pre=net1(tar1)
pre
```




    tensor([[0.7201]], grad_fn=<AddmmBackward>)




```python
pre.detach().numpy()
```




    array([[0.72008437]], dtype=float32)




```python
deScaler.inverse_transform(pre.detach().numpy())
```




    array([[69334.53]], dtype=float32)


