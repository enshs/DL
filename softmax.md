# softmax
_______________________

독립변수를 3개의 클래스를 가진 반응변수에 매핑시키는 모델을 생성합니다. 즉, 데이터는 다음의 구조를 가집니다. 

|x|y|
|---|---|
|1.5|0|
|...|...|

클래스 0, 1, 2를 각각 A, B, C로 나타내면 각각에 부여되는 가중치를 $W_A, W_B, W_C$로 표시하여 x에 대한 예측치를 계산할 수 있습니다. 

$$\begin{aligned}W_A \cdot x &= H_A\\W_B \cdot x &= H_B\\W_c \cdot x &= H_c\end{aligned}$$

간단한 예로 다음의 경우에 대해서만 예측치를 계산해 봅니다. 

x=1.5, y=0


```python
import numpy as np
import torch
```


```python
W1=torch.rand(3)
W1
```




    tensor([0.5237, 0.8197, 0.8847])




```python
x=1.5
haty1=W1*x
haty1
```




    tensor([0.7856, 1.2295, 1.3271])



위 결과 haty는 x가 label의 각 클래스 A, B, C라는 가정에 대한 예측입니다. 위 값들이 [0, 1] 구간내에 포함하도록 정규화 할 수 있습니다. 이것은 feature가 A, B, C에 매핑될 확률로 간주할 수 있습니다. 
$$\begin{aligned}S(\hat{y_i})&=P_i\\&=\frac{e^{\hat{y_i}}}{\sum^k_{j=1}e^{\hat{y_i}}}\\ k:& \text{클래스 갯수} \end{aligned}$$

위 계산결과의 총합은 1이 됩니다. 


```python
re1=haty1/torch.sum(haty1)
re1
```




    tensor([0.2351, 0.3679, 0.3971])




```python
torch.sum(re1)
```




    tensor(1.)



위 결과  중 독립변수인 1.5는 3 번째 클래스에 포함될 확률이 가장 큽니다. 그러므로 위 결과의 예측은 2이며 실제값은 0입니다. 위 결과와 실제값과의 차이를 최소로 하는 모델을 구현해야 합니다. 차이가 최소림을 나타낼 지료 즉, 비용함수가 필요합니다. 선형함수에서 사용한 mse를 사용하여 다음과 같습니다. 
$$(2-0)^2=4$$
위 과정을 다시 실시해 봅니다. 


```python
W2=torch.rand(3)
haty2=w=W2*x
re2=haty2/torch.sum(haty2)
re2
```




    tensor([0.3553, 0.3280, 0.3167])



위 결과로 클래스 2에 포함됩니다. 이 경우 비용은 다음과 같습니다.
$$(1 − 0)^2 = 1$$
위의 두 비용 4, 1은 클래스 선택의 오류에 기인합니다. 이 결과는 클래스 2와 클래스 1의 선택에 대한 그 차이를 나타냅니다. 즉, 실제값 0은 2보다 1에 더 가깝다라는 의미를 나타냅니다. 그러나 클래스 0, 1, 2는 분류를 위한 것으로 순위를 나타내는 것이 아닙니다. 그러므로 비용 4와 1의 크기는 의미를 가지지 않아야 합니다. 이러한 문제를 해결하기 위해 반응변수를 즉, 클래스를 원-핫 인코딩(one-hot encoding)으로 전환합니다. 원-핫인코딩은 참=1, 거짓=0의 값을 부여합니다. 즉 클래스 3개는 다음과 같이 나타낼 수 있습니다.

|클래스|0|1|2|
|---|---|---|---|
|거짓|0|0|0|
|참|1|1|1|

0클래스에 포함되는 y는 [1, 0, 0] 으로 전환합니다. 이 값과 예측결과인 확률의 곱의 크기로 비용을 계산합니다. 즉, 다음과 같습니다.
$$\begin{aligned}\text{cost}&=(y \cdot log(P))\\&=-(1 \cdot log(P_1)+ 0 \cdot log(P_2)+ 0 \cdot log(P_1)\\ &=\sum^3_{i=1}-\text{Class}_i \cdot log(P_i)\end{aligned}$$
즉, 관측치에 해당하는 클래스에 예측되는 확률만을 고려하는 것입니다.이 비용함수를 크로스 엔트로피(cross-entropy)함수라고 합니다. 


```python
yOH=torch.FloatTensor([1,0,0])
cost1=-torch.sum(yOH*torch.log(re1))
cost1
```




    tensor(1.4479)




```python
cost2=-torch.sum(yOH*torch.log(re2))
cost2
```




    tensor(1.0349)



위 과정은 다음과 같이 일반화합니다. 3개이상의 분류를 위해서는 다음과 같이  로지스틱 분류를 사용하여 one or nothing 방법을 적용합니다. 예를 들어 자료의 독립변수에 대응하는 종속변수가 A, B, C 라고 하면 로지스틱 분류에 의해 다음과 같이 구분하여 분석할 수 있습니다.

1. A or not

$\left[\begin{matrix} W_{A1} &W_{A2}&W_{A3} \end{matrix} \right] \left[ \begin{matrix} x_1\\x_2\\x_3 \end{matrix}\right] = \left[\begin{matrix} W_{A1}x_1+W_{A2}x_2+W_{A3}x_3 \end{matrix}\right]=\hat{y_A} $ 

2. B or not 
 
$\left[\begin{matrix} W_{B1} &W_{B2}&W_{B3} \end{matrix} \right] \left[ \begin{matrix} x_1\\x_2\\x_3 \end{matrix}\right] = \left[\begin{matrix} W_{B1}x_1+W_{B2}x_2+W_{B3}x_3 \end{matrix}\right]=\hat{y_B}$   

3. C or not 

$\left[\begin{matrix} W_{C1} &W_{C2}&W_{C3} \end{matrix} \right] \left[ \begin{matrix} x_1\\x_2\\x_3 \end{matrix}\right] = \left[\begin{matrix} W_{C1}x_1+W_{C2}x_2+W_{C3}x_3 \end{matrix}\right]=\hat{y_C}$ 

위의 각각의 가설함수는 다음과 같이 결합하여 나타낼 수 있습니다.   
$$\left[ \begin{matrix} W_{ A1 } & W_{ A2 } & W_{ A3 } \\ W_{ B1 } & W_{ B2 } & W_{ B3 } \\ W_{ C1 } & W_{ C2 } & W_{ C3 } \end{matrix} \right] \left[ \begin{matrix} x_{ 1 } \\ x_{ 2 } \\ x_{ 3 } \end{matrix} \right] =\left[ \begin{matrix} W_{ A1 }x_{ 1 }+W_{ A2 }x_{ 2 }+W_{ A3 }x_{ 3 } \\ W_{ B1 }x_{ 1 }+W_{ B2 }x_{ 2 }+W_{ B3 }x_{ 3 } \\ W_{ C1 }x_{ 1 }+W_{ C2 }x_{ 2 }+W_{ C3 }x_{ 3 } \end{matrix} \right] =\left[ \begin{matrix} \hat { y_{ A } }  \\ \hat { y_{ B } }  \\ \hat { y_{ C } }  \end{matrix} \right]$$
위 식에서 W는 <mark>(변수의 수, 클래스의 수)</mark>의 차원으로 무작위로 초기값을 설정합니다
위의 연산으로부터 예측치를 [0,1] 사이의 값으로 변환하기 위해 각 값들을 전체의 값으로 나누어 고려합니다. 이 연산자를 소프트맥스(softmax) 라고 합니다.    
$$\begin{equation}
	S(\hat{y_i})=\frac{exp(\hat{y_i})}{\sum_i exp(\hat{y_i})}
\end{equation}$$
pytorch.functional 모듈의 softmax(x, dim=0)을 사용합니다. 이 함수에서 dim은 객체 x에서 변수의 차원입니다. 
위 연산 결과인  $S(\hat{y_i})$는 그 범위를 표준화(정규화)시킨 y의 예측값 즉, $\hat{y}$ 이며 각 경우의 확률로 고려될 수 있습니다.  예측된 각 확률과 원-핫 인코딩된 실제값(반응변수)과의 곱으로 비용을 계산합니다. 즉, 비용함수로  크로스 엔트로피(cross-entropy)함수를 적용합니다. 
$$\begin{equation}
	\begin{aligned}\text{cost}&=-(y \times  \log(P))\\&=-(1 \cdot  \log(P_1) +  0 \cdot \log( P_2) + 0 \times   \log(P_3))\\&=\sum^3_{j=1} yOH_i \cdot \log(P_j)\\ yOH: &\text{one-hot encoding에 의한 관찰치} \end{aligned} 
\end{equation}$$

위 로그함수를 근거로 Cross-entropy 비용함수의 적합성을 알아봅니다.    
예를들어 L(=y)이 다음일 때 예측치 $\hat{y_1}$과 $\hat{y_2}$에 대한 Cross-entropy  함수의 결과는 다음과 같습니다.   
$$y=\left[\begin{matrix}0\\1\end{matrix}\right], \quad \hat{y_1}=\left[\begin{matrix}0\\1\end{matrix}\right] \quad \hat{y_2}=\left[\begin{matrix}1\\0\end{matrix}\right]$$   
$$\begin{aligned}&\begin{aligned}y\times (-log(\hat { y_{ 1 } } ))&=\left[ \begin{matrix} 0 \\ 1 \end{matrix} \right] \times -log\left( \left[ \begin{matrix} 0 \\ 1 \end{matrix} \right]  \right) \\&=\left[ \begin{matrix} 0 \\ 1 \end{matrix} \right]  \times \left[ \begin{matrix} \infty \\ 0 \end{matrix} \right] \\&=\left[ \begin{matrix} 0 \\ 0 \end{matrix} \right]\\&=0+0\\&=0 \end{aligned}\\ &\because \; -log(0)  \approx \infty ,\; -log(1) \approx 0 \end{aligned}$$  
$$\begin{aligned}&\begin{aligned} y\times (-log(\hat { y_{ 2} } ))&=\left[ \begin{matrix} 0 \\ 1 \end{matrix} \right] \times -log\left( \left[ \begin{matrix} 1 \\ 0 \end{matrix} \right]  \right) \\ &=\left[ \begin{matrix} 0 \\ 1 \end{matrix} \right] \times \left[ \begin{matrix} 0 \\ \infty  \end{matrix} \right] \\ &=\left[ \begin{matrix} 0 \\ \infty  \end{matrix} \right]\\ &=0+\infty \\&=\infty \end{aligned}\\& \because \; -log(0)\approx \infty ,\; -log(1)\approx 0\end{aligned}$$

그러므로 실측치와 일치하지 않은 경우 cross-entropy함수값은 매우 증가하므로 이 함수를 포함하는 비용함수 역시 매우 증가합니다.

다음은 iris 데이터에 대한 softmax회귀를 실행합니다. 이 데이터는 sklearn 모듈에 포함되므로 다음과정으로 호출할 수 있습니다. 


```python
from sklearn.datasets import load_iris
da=load_iris()
da['data'][:3]
```




    array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2]])




```python
da['target'][:3]
```




    array([0, 0, 0])



위 결과와 같이 da['data']가 독립변수, da['target']는 반응변수로 구성되었으며 데이터를 학습과 검정 그룹으로 구분하기 위해 klearn.model_selection 모듈의 train_test_split() 함수를 사용합니다.


```python
#학습, 검정 데이터로 분리
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte=train_test_split(da['data'], da['target'])
xtr[:3]
```




    array([[7.7, 2.6, 6.9, 2.3],
           [4.9, 3.1, 1.5, 0.2],
           [7. , 3.2, 4.7, 1.4]])




```python
da['target']
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
import pandas as pd
target_pd=pd.DataFrame(da['target'])
target_pd.describe()
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.819232</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
xtr.shape
```




    (112, 4)



위 자료들은 numpy 배열형이므로 tensor로 전환합니다. 


```python
xtr1, xte1, ytr1, yte1=map(torch.tensor, (xtr, xte, ytr, yte))
print(xtr1.shape, xte1.shape)
```

    torch.Size([112, 4]) torch.Size([38, 4])



```python
ytr1
```




    tensor([2, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 2, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 0, 2,
            1, 1, 2, 2, 1, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 2, 0, 1, 0, 2,
            0, 1, 2, 1, 1, 1, 1, 1, 0, 2, 1, 2, 0, 1, 1, 2, 1, 1, 2, 2, 0, 1, 0, 2,
            0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 1, 0,
            2, 0, 0, 1, 1, 2, 2, 1, 0, 1, 0, 1, 0, 1, 0, 2])




```python
import torch.nn.functional as F 
ytrOH=F.one_hot(ytr1, 3)
yteOH=F.one_hot(yte1, 3)
ytrOH[:3]
```




    tensor([[0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]])



위 결과에 의하면 label의 클래스는 0,1,2로 3개이고 4개의 feature로 구성되어 있습니다. 그러므로 가중치의 차원은 (4, 3)이 됩니다. 이를 근거로 먼저 모델에 사용될 가중치 객체를 초기화하고 모델을 구성합니다. 

모델은 가중치를 고려하여 softmax함수를 적용하고 비용함수를 선언합니다. 이 결과를 기준으로 가중치를 개선하기 위해 과정을 반복합니다.

softmax 함수는 torch.nn.fucntional.sofmax(tensor, dim)을 사용합니다. 이 경우 
- tensor: xtr @ W +b
- 위 행렬곱은 xtr의 각행은 W의 각 열과의 곱으로 데이터 하나의 인스턴스의 예측치를 생성합니다. 결과적으로 하나의 인스턴스의 softmax의 계산은 한 행의 열로 구성되어 있는 각 클래스들의 합을 기준으로 계산됩니다. 열의 차원은 1이므로 위 함수의 dim=1이 됩니다. 

위 softmax() 함수내의 계산은 다음과 같이 이루어집니다. 
$$\begin{aligned}&\left[\begin{matrix}x_{11}&\cdots&x_{1p}\\x_{21}&\cdots&x_{2p}\\\vdots& \ddots & \vdots\\x_{n1}&\cdots&x_{np}\end{matrix}\right]
	\left[\begin{matrix}W_{A1}&\cdots&W_{N1}\\\vdots&\ddots&\vdots\\W_{AP}&\cdots&W_{NP}\end{matrix}\right]+b\\&=\left[\begin{matrix} (x_{11}W_{A1}+\cdots+x_{1p}W_{AP})+b&\cdots &(x_{11}W_{N1}+\cdots+x_{1p}W_{NP})+b\\\vdots&\ddots&\vdots\\(x_{n1}W_{A1}+\cdots+x_{np}W_{AP})+b&\cdots &(x_{n1}W_{N1}+\cdots+x_{np}W_{NP})+b \end{matrix} \right] \end{aligned}$$
위 결과의 각 행이 인스턴스 각각의 예측치가 됩니다. 즉, 확률로 전환하는 softmax()함수는 각 행의 합에 대해 각 열의 크기를 고려합니다. 그러므로 softmax() 함수의 계산 차원은 열이 됩니다(dim=1). 
softmax모델은 다음과 같습니다.  
```
model=F.softmax(data.matmul(W)+b, dim=1)
```


```python
#가중치와 편차 초기화
W=torch.rand((4, 3), requires_grad=True, dtype=torch.double)
b=torch.rand(1, requires_grad=True, dtype=torch.double)
# optimizer 설정
optimizer = torch.optim.SGD([W, b], lr=0.1)
```


```python
W.dtype
```




    torch.float64




```python
def model(x, W, b):
    return(F.softmax(x.matmul(W)+b, dim=1))
```


```python
n=10000
for epoch in range(n+1):
    #순전파
    pre=model(xtr1, W, b)
    cost=(-ytrOH*torch.log(pre)).sum(dim=1).mean()
    #1000번마다 출력
    if epoch %1000==0:
        print(f'Epoch:{epoch}, cost:{np.round(cost.item(), 4)}')
    #역전파
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
```

    Epoch:0, cost:0.0634
    Epoch:1000, cost:0.0627
    Epoch:2000, cost:0.0621
    Epoch:3000, cost:0.0616
    Epoch:4000, cost:0.0612
    Epoch:5000, cost:0.0609
    Epoch:6000, cost:0.0606
    Epoch:7000, cost:0.0604
    Epoch:8000, cost:0.0602
    Epoch:9000, cost:0.06
    Epoch:10000, cost:0.0598


위 연산 결과의 모델에 데이터를 적용한 예측치의 각행의 합은 1이 됩니다. 


```python
W, b
```




    (tensor([[ 2.7902,  2.0897, -3.3427],
             [ 4.4736,  0.8052, -3.3952],
             [-5.0204,  0.7631,  6.0751],
             [-2.4827, -3.9221,  7.9394]], dtype=torch.float64, requires_grad=True),
     tensor([0.4782], dtype=torch.float64, requires_grad=True))




```python
xtr_pre=model(xtr1, W, b)
xtr_pre[:3].sum(dim=1)
```




    tensor([1.0000, 1.0000, 1.0000], dtype=torch.float64, grad_fn=<SumBackward1>)



행 단위로 가장 큰 확률의 열에 해당하는 클래스가 최종 예측 결과가 됩니다. 행단위로 가장 큰 값을 찾기 위해 torch.argmax(data, dim) 함수를 적용합니다.


```python
torch.argmax(xtr_pre, dim=1)
```




    tensor([2, 0, 1, 1, 1, 1, 2, 2, 1, 0, 2, 2, 2, 0, 2, 2, 1, 1, 1, 2, 0, 1, 0, 2,
            1, 1, 2, 2, 1, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 1, 0, 2, 0, 1, 0, 2,
            0, 1, 2, 1, 1, 1, 1, 1, 0, 2, 1, 2, 0, 1, 1, 2, 1, 1, 2, 2, 0, 1, 0, 2,
            0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 1, 0,
            2, 0, 0, 1, 1, 2, 2, 1, 0, 1, 0, 1, 0, 1, 0, 2])



위 결과와 실측값 (ytr1)이 같으면 True(1), 다르면 False(0) 을 부여하여 모두 True인 경우로 나누어주면 이 예측의 정확도를 결정할 수 있습니다.


```python
torch.sum(torch.argmax(xtr_pre, dim=1)==ytr1).item()/len(ytr1)
```




    0.9821428571428571



높은 정확도를 보입니다. 검정 데이터에 대해 위 과정을 적용합니다.


```python
xte_pre=model(xte1, W, b)
torch.sum(torch.argmax(xte_pre, dim=1)==yte1).item()/len(yte1)
```




    0.9210526315789473



학습데이터와 검정 데이터 사이에 차이가 존재합니다. 즉, 과적합(over estimation)의 가능성이 있습니다. 과적합을 감소시키기 위한 방법을 검토해야 합니다.

위 코드에서 모델에 softmax()함수를 적용하고 비용함수에서 그 결과에 대한 log 값의 전환을 위해 torch.log()함수를 사용했습니다. pytorch는 이 두 함수를 결합한 torch.nn.functional.log_softmax()를 제공합니다. 다음 코드는 이 두 함수를 비교한 것입니다.  


```python
com1=torch.log(F.softmax(xtr1[:3,:].matmul(W)+b, dim=1))
com1
```




    tensor([[-3.2847e+01, -1.1184e+01, -1.3895e-05],
            [-1.6316e-03, -6.4190e+00, -3.5719e+01],
            [-8.5248e+00, -2.4891e-04, -9.8957e+00]], dtype=torch.float64,
           grad_fn=<LogBackward>)




```python
com2=F.log_softmax(xtr1[:3,:].matmul(W)+b, dim=1)
com2
```




    tensor([[-3.2847e+01, -1.1184e+01, -1.3895e-05],
            [-1.6316e-03, -6.4190e+00, -3.5719e+01],
            [-8.5248e+00, -2.4891e-04, -9.8957e+00]], dtype=torch.float64,
           grad_fn=<LogSoftmaxBackward>)



torch.nn.functional.log_softmax()를 적용한 비용함수는 다음과 같습니다. 
```
cost=(-y*torch.nn.functional.log_softmax(x)).sum(dim=1).mean()
```
위 연산의 y는 one-hot encoding으로 전환된 객체이어야 합니다. 그러나 pytorch는 y를 one-hot으로 전환하지 않은 정수값 그대로 전달하고 비용을 계산하는 함수를 제공합니다. 이 함수는 각 인스턴스의 총 합의 평균을 동시에 계산합니다.  
```
cost=F.nll_loss(F.log_softmax(x), y)
```


```python
(-ytrOH[:3]*com2).sum(dim=1).mean()
```




    tensor(0.0006, dtype=torch.float64, grad_fn=<MeanBackward0>)




```python
F.nll_loss(com2, ytr1[:3])
```




    tensor(0.0006, dtype=torch.float64, grad_fn=<NllLossBackward>)



위 함수는 null_loss()는 softmax()를 인수로 받습니다. 이 두 함수를 결합한 cross_entropy()함수를 사용할 수 있습니다. 즉, 다음과 같습니다. 

*F.log_softmax()+F.nll_loss()=F.cross_entropy()*
- F.cross_entropy()에 전달되는 예측모형은 선형모형입니다. 
- 이 함수에 전달하는 y는 one-hot 인코딩 형태가 아닌 정수형이어야 합니다
  
위 함수는 비용함수 내에 softmax()함수를 포함하고 있기 때문에 전달되는 인수는 가중치와 편차를 고려한 선형모형입니다.


```python
F.cross_entropy(xtr1[:3,:].matmul(W)+b, ytr1[:3])
```




    tensor(0.0006, dtype=torch.float64, grad_fn=<NllLossBackward>)



선형모형 nn.Linear()과 F.cross_entropy()를 적용하여 모델을 다시 구현합니다. 


```python
xtr1.dtype, ytr1.dtype
```




    (torch.float64, torch.int64)




```python
model2=torch.nn.Linear(4, 3)
optimizer=torch.optim.SGD(model2.parameters(), lr=0.1)
n=10000
for epoch in range(n+1):
    pre=model2(xtr1.to(torch.float))
    cost=F.cross_entropy(pre, ytr1.to(torch.int64))
    if epoch % 1000 ==0:
        acc=torch.mean((torch.argmax(model2(xtr1.to(torch.float)), dim=1)==ytr1).float())
        print(f'Epoch:{epoch}, cost:{np.around(cost.item(), 4)}, accuracy: {np.around(acc.item(), 3)}')
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
```

    Epoch:0, cost:1.1687, accuracy: 0.527
    Epoch:1000, cost:0.1139, accuracy: 0.982
    Epoch:2000, cost:0.0819, accuracy: 0.991
    Epoch:3000, cost:0.0688, accuracy: 0.991
    Epoch:4000, cost:0.0612, accuracy: 0.991
    Epoch:5000, cost:0.0562, accuracy: 0.991
    Epoch:6000, cost:0.0524, accuracy: 0.991
    Epoch:7000, cost:0.0495, accuracy: 0.991
    Epoch:8000, cost:0.0471, accuracy: 0.991
    Epoch:9000, cost:0.0451, accuracy: 0.991
    Epoch:10000, cost:0.0433, accuracy: 0.991



```python
#검정데이터의 정확도 
torch.mean((torch.argmax(model2(xte1.to(torch.float)),dim=1)==yte1).float())
```




    tensor(0.9211)


