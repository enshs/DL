```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import FinanceDataReader as fdr
```


```python
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
```

주가자료(삼성전자)에 대해 실시해 봅니다. 
1. 데이터 호출 
 자료는 google finantial로 부터 호출된 자료이며 비거래일은 0으로 표시됩니다. 이것은 이후 계산에서 문제를 일으키므로 삭제 또는 다른 값으로 변환이 필요합니다. pandas객체.replace() 메서드를 사용하여 그 값들을 변환할 필요가 있습니다. 여기서는 전날의 값들로 0을 대신합니다. 즉, 이 함수의 인수중 `method='ffill'`을 설정합니다. 
2. 1일간의 변화 생성(DataFrame.pct_change())
 이 결과는 1번의 첫날의 자료가 na이 됩니다. 그러므로 이 부분을 삭제하고 numpy.array 형으로 변환합니다. (이후에 사용할 sklearn과 pytorch의 자료는 numpy.array 자료이어야 합니다.)
3. 2의 결과는 독립변수(feature)이므로 각각에 대응하는 label(반응변수)를 생성합니다. 
  여기에서는 1번의 자료로부터 `Close`의 목록화하여 사용합니다. pandas.cut()을 사용합니다. 
  물론 2의 결과와 같은 차원으로 조정하며 numpy array 형으로 변환합니다.   
4. feature, lable에 대해 학습데이터와 검증데이터를 구분합니다. sklearn.model_selection.train_test_split()함수를 사용합니다.
5. 데이터를 torch.tensor 형으로 전환합니다. map()함수를 사용합니다. 
xtr1, xte1, ytr1, yte1=map(torch.tensor,(xtr, xte, ytr, yte)) : 이렇게 전환할 경우 각각의 데이터 형태(dtype)은 numpy의 자료형을 상속받습니다. 
6. pytorch의 nn.Linear()함수를 적용하기 위해서는 데이터가 독립변수의 자료형은 float32 형태이어야 합니다.<br> tensor객체.to(detyle)을 사용합니다. 
   softmax를 적용할 경우 lable은 정수형이어야 합니다. tensor객체.to(detyle)을 사용합니다.
7. 배치 크기를 64로 하기위해 tensorDataset()과 DataLoader()함수를 적용합니다. 


```python
st=pd.Timestamp(2010,1, 1)
et=pd.Timestamp(2021, 10, 21)
sam=fdr.DataReader('005930', st, et)
sam.head()
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
    <tr>
      <th>2010-01-07</th>
      <td>16820</td>
      <td>16820</td>
      <td>16260</td>
      <td>16260</td>
      <td>443237</td>
      <td>-0.033294</td>
    </tr>
    <tr>
      <th>2010-01-08</th>
      <td>16400</td>
      <td>16420</td>
      <td>16120</td>
      <td>16420</td>
      <td>295798</td>
      <td>0.009840</td>
    </tr>
  </tbody>
</table>
</div>



값이 0인 부분을 이전의 값으로 대치합니다. 


```python
sam1=sam.replace(0, method="ffill")
```

1일간의 변화량을 산출합니다.이 자료는 DataFrame 형으로 numpy array 형으로 변환합니다. 


```python
samChg=sam1.pct_change(periods=1, fill_method='ffill')
X=samChg.dropna()
X.head(3)
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
      <th>2010-01-05</th>
      <td>0.028643</td>
      <td>0.024722</td>
      <td>0.018750</td>
      <td>0.016069</td>
      <td>1.337178</td>
      <td>0.283931</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>0.003632</td>
      <td>0.014475</td>
      <td>0.013497</td>
      <td>0.023114</td>
      <td>-0.177862</td>
      <td>0.438424</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>0.014475</td>
      <td>0.000000</td>
      <td>-0.015738</td>
      <td>-0.033294</td>
      <td>-0.035928</td>
      <td>-2.440391</td>
    </tr>
  </tbody>
</table>
</div>




```python
samChg.shape, X.shape
```




    ((2912, 6), (2911, 6))



자료를 표준화합니다. 


```python
scaler=MinMaxScaler()
Xnor=scaler.fit(X).transform(X)
Xnor
```




    array([[0.58461909, 0.61370607, 0.57376047, 0.52104608, 0.03276656,
            0.48858952],
           [0.46610913, 0.54691315, 0.5444871 , 0.55911492, 0.00948003,
            0.4896474 ],
           [0.51748889, 0.45255551, 0.38156898, 0.25431036, 0.0116616 ,
            0.4699349 ],
           ...,
           [0.48939851, 0.51746274, 0.56590887, 0.46500455, 0.00767969,
            0.50714861],
           [0.44220687, 0.45255551, 0.45344242, 0.41125368, 0.01325431,
            0.4746914 ],
           [0.42209111, 0.41583135, 0.45339732, 0.42652857, 0.0133334 ,
            0.4820901 ]])



라벨(반응변수)는 `Close 변화율`을 목록화하여 사용합니다. 역시 numpy array 형으로 변환합니다. 


```python
binsize=30
y, group=pd.cut(X["Close"], bins=binsize, labels=np.arange(0, binsize), retbins=True)
ynp=np.array(y)
ynp
```




    array([15, 16,  7, ..., 13, 12, 12])




```python
group
```




    array([-0.08054221, -0.07418838, -0.06801961, -0.06185084, -0.05568207,
           -0.04951331, -0.04334454, -0.03717577, -0.031007  , -0.02483824,
           -0.01866947, -0.0125007 , -0.00633193, -0.00016317,  0.0060056 ,
            0.01217437,  0.01834314,  0.0245119 ,  0.03068067,  0.03684944,
            0.04301821,  0.04918697,  0.05535574,  0.06152451,  0.06769328,
            0.07386204,  0.08003081,  0.08619958,  0.09236835,  0.09853711,
            0.10470588])




```python
Xnor.shape, ynp.shape
```




    ((2911, 6), (2911,))



feature와 label은 1일의 시차를 가집니다. 즉, 1일 전의 독립변수로 현재를 예측합니다.독립변수의 마지막 행 자료는 최종 예측을 위해 사용됩니다. 


```python
Xf=Xnor[:-1,:]
yl=ynp[1:]
new=Xnor[-1,:]
```

학습데이터와 검증 데이터로 구분합니다. 


```python
xtr, xte,ytr,yte=train_test_split(Xf, yl, test_size=0.2, random_state=1) 
xtr.shape, xte.shape
```




    ((2328, 6), (582, 6))



위의 데이터는 np.array 형태이므로 tensor 형으로 변환 

아래의 과정에서 dtype은 array와 같음


```python
xtr1, xte1, ytr1, yte1, newT=map(torch.tensor,(xtr, xte, ytr, yte, new))
```


```python
print(newT)
xtr1.shape, xtr1.dtype, ytr1.shape, ytr1.dtype
```

    tensor([0.4221, 0.4158, 0.4534, 0.4265, 0.0133, 0.4821], dtype=torch.float64)





    (torch.Size([2328, 6]), torch.float64, torch.Size([2328]), torch.int64)



nn.Linear()함수를 적용하기 위해서는 인수는 float32형태, softmax에서 label은 정수형(int64)이므로 다음 코드와 같이 변환합니다. 


```python
xtr1, xte1, newT=xtr1.to(torch.float), xte1.to(torch.float), newT.to(torch.float)
```


```python
xtr1.dtype, ytr1.dtype, newT.dtype
```




    (torch.float32, torch.int64, torch.float32)



배치 크기를 500로 하기위해 tensorDataset()과 DataLoader()함수를 적용합니다. 


```python
bs=500
trDs = TensorDataset(xtr1, ytr1)
trDl = DataLoader(trDs, batch_size=bs)
```


```python
teDs=TensorDataset(xte1, yte1)
teDl=DataLoader(teDs, batch_size=bs)
```

layer와 순전파 계산을 위한 함수 즉, 모델을 설정합니다. 여기서는 1개층의 선형모델인 torch.nn.Linear()함수를 적용합니다. 이 함수의 인수는 입력되는 데이터(feature)의 열과 출력(label) 열입니다. 그러므로 다음 코드와 같이 (6, 15)가 됩니다.<br>
비용함수로 torch.nn.functional.cross_entropy()을 사용합니다. 이 함수의 인수는 모델에서 계산된 결과와 관측치 값(label)입니다. 또한 이 함수는 sofmax()함수를 포함하므로 선형모델을 사용할 수 있습니다. 


```python
class stockLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin=nn.Linear(6, binsize)
    def forward(self, x):
        return self.lin(x)
    
loss_func=F.cross_entropy
```

최적화를 설정합니다. 


```python
lr=1e-3
model=stockLogistic()
opt=optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

다음으로 학습 루프를 작성합니다. 


```python
loopN=10000
for epoch in range(loopN+1):
    for xb, yb in trDl:
        pre=model(xb)
        loss=F.cross_entropy(pre, yb)
        opt.zero_grad()
        loss.backward()
        opt.step() 
        
    model.eval()
    with torch.no_grad():
        valid_loss=sum(F.cross_entropy(model(xb), yb) for xb, yb in teDl)
    if epoch%1000==0:
            print(f'Epoch:{epoch}, loss:{np.round(loss.item(),4)}, valid_loss:{np.round(valid_loss.item(),4)}')
```

    Epoch:0, loss:3.5787, valid_loss:7.1634
    Epoch:1000, loss:2.47, valid_loss:5.0654
    Epoch:2000, loss:2.4413, valid_loss:5.0211
    Epoch:3000, loss:2.4321, valid_loss:5.0077
    Epoch:4000, loss:2.4273, valid_loss:5.0006
    Epoch:5000, loss:2.424, valid_loss:4.9959
    Epoch:6000, loss:2.4215, valid_loss:4.9927
    Epoch:7000, loss:2.4195, valid_loss:4.9904
    Epoch:8000, loss:2.4178, valid_loss:4.9887
    Epoch:9000, loss:2.4163, valid_loss:4.9875
    Epoch:10000, loss:2.415, valid_loss:4.9867



```python
w, b=model.parameters()
w.shape, b.shape
```




    (torch.Size([30, 6]), torch.Size([30]))




```python
torch.sum(F.softmax(model(xtr1), dim=1), dim=1)
```




    tensor([1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
           grad_fn=<SumBackward1>)




```python
#학습데이터에 대한 평가 
torch.mean((torch.argmax(model(xtr1), dim=1)==ytr1).float())
```




    tensor(0.1899)




```python
#검정데이터에 대한 평가 
torch.mean((torch.argmax(model(xte1), dim=1)==yte1).float())
```




    tensor(0.1856)



최종 대상 newT에 대한 예측을 실시합니다. 현재 newT는 1차원입니다. 그러나 Linear 모델에서 인수는 2차원구조를 가져야 합니다. 그러므로 차원변경을 먼저 실시한 후에 에측을 실행합니다. 


```python
newT1=newT.reshape(1, -1)
newT1
```




    tensor([[0.4221, 0.4158, 0.4534, 0.4265, 0.0133, 0.4821]])




```python
preIndex=torch.argmax(model(newT1))
preInterval=group[preIndex.item():(preIndex.item()+2)]
sam["Close"][-1]+preInterval*sam["Close"][-1]
```




    array([70188.54579832, 70621.59327731])



위의 softmax 모델은 비용함수로 F.cross_entry()를 사용했습니다. 이 함수는 내부에 softmax 연산이 포함되어 있으므로 layer에서 선형모델의 적용이 가능합니다. 

다음의 경우는 모델 설정 단계에서 softmax를 적용하는 것입니다. 이 경우 label은 one-hot encoding을 실시합니다. 이 모형에서는 배치를 사용하지 않았습니다. 


```python
ytr1OH=F.one_hot(ytr1, binsize)
yte1OH=F.one_hot(yte1, binsize)
ytr1OH[0]
```




    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0])




```python
xtr1.shape, ytr1OH.shape
```




    (torch.Size([2328, 6]), torch.Size([2328, 30]))




```python
#가중치와 편차 초기화
W=torch.rand((xtr1.shape[1], ytr1OH.shape[1]), requires_grad=True, dtype=torch.float)
b=torch.rand(1, requires_grad=True, dtype=torch.float)
# optimizer 설정
optimizer = torch.optim.Adam([W, b], lr=0.01)
```


```python
def model1(x, W, b):
    return(F.softmax(x.matmul(W)+b, dim=1))
```


```python
n=10000
for epoch in range(n+1):
    #순전파
    pre=model1(xtr1, W, b)
    cost=(-ytr1OH*torch.log(pre)).sum(dim=1).mean()
    #1000번마다 출력
    if epoch %1000==0:
        print(f'Epoch:{epoch}, cost:{np.round(cost.item(), 4)}')
    #역전파
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
```

    Epoch:0, cost:3.5847
    Epoch:1000, cost:2.3887
    Epoch:2000, cost:2.3833
    Epoch:3000, cost:2.3799
    Epoch:4000, cost:2.3766
    Epoch:5000, cost:2.3733
    Epoch:6000, cost:2.3704
    Epoch:7000, cost:2.3681
    Epoch:8000, cost:2.3665
    Epoch:9000, cost:2.3654
    Epoch:10000, cost:2.3647



```python
torch.sum(model1(xtr1, W, b), dim=1)
```




    tensor([1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
           grad_fn=<SumBackward1>)




```python
torch.mean((torch.argmax(model1(xtr1, W, b),dim=1)==ytr1).float())
```




    tensor(0.1937)




```python
torch.mean((torch.argmax(model1(xte1, W, b),dim=1)==yte1).float())
```




    tensor(0.1838)




```python
preIndex1=torch.argmax(model1(newT1, W, b))
preInterval1=group[preIndex1.item():(preIndex1.item()+2)]
sam["Close"][-1]+preInterval1*sam["Close"][-1]
```




    array([70188.54579832, 70621.59327731])


여러층의 모형을 생성할 수 있는 `nn.sequential()`과 비선형함수인 `nn.ReLU()`를 사용하여 다층모형을 구현해 봅니다. 

```(python)
class stockLogistic3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, binsize), nn.ReLU())
    def forward(self, x):
        return self.layer(x)
    
model3=stockLogistic3()

lr=1e-3
opt=optim.Adam(model3.parameters(), lr=lr)

loopN=10000
for epoch in range(loopN+1):
    pre=model3(xtr1)
    loss=F.cross_entropy(pre, ytr1)    
    opt.zero_grad()
    loss.backward()
    opt.step() 
    
    model.eval()
    with torch.no_grad():
        valid_loss=F.cross_entropy(model3(xte1), yte1)
     
    if epoch%1000==0:
            print(f'Epoch:{epoch}, loss:{np.round(loss.item(),4)}, val_loss:{np.round(valid_loss.item(), 4)}')
```

```
Epoch:0, loss:2.7353, val_loss:2.8229
Epoch:1000, loss:2.7309, val_loss:2.8284
Epoch:2000, loss:2.728, val_loss:2.8323
Epoch:3000, loss:2.7253, val_loss:2.8373
Epoch:4000, loss:2.7234, val_loss:2.8414
Epoch:5000, loss:2.7218, val_loss:2.844
Epoch:6000, loss:2.7204, val_loss:2.8451
Epoch:7000, loss:2.7191, val_loss:2.8464
Epoch:8000, loss:2.7181, val_loss:2.8482
Epoch:9000, loss:2.7171, val_loss:2.8481
Epoch:10000, loss:2.7163, val_loss:2.849
```


학습 데이터에 대한 정확도

```(python)
torch.mean((torch.argmax(model3(xtr1),dim=1)==ytr1).float())
```
```
tensor(0.2294)
```

검증데이터에 대한 정확도 

```(python)
torch.mean((torch.argmax(model3(xte1),dim=1)==yte1).float())
```
```
tensor(0.1684)
```

최종 변수(newT)에 대한 예측

```(python)
preIndex3=torch.argmax(model3(newT1))
preInterval3=group[preIndex3.item():(preIndex3.item()+2)]
sam["Close"][-1]+preInterval3*sam["Close"][-1]
```
```
array([70188.54579832, 70621.59327731])
```
