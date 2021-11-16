```python
import numpy as np
import pandas as pd
```


```python
df_original=pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0]
df_original.iloc[1,:]
```




    회사명                    DSR
    종목코드                155660
    업종             1차 비철금속 제조업
    주요제품                합섬섬유로프
    상장일             2013-05-15
    결산월                    12월
    대표자명                   홍석빈
    홈페이지    http://www.dsr.com
    지역                   부산광역시
    Name: 1, dtype: object




```python
df=df_original.iloc[:, :3]
df.head()
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
      <th>회사명</th>
      <th>종목코드</th>
      <th>업종</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DRB동일</td>
      <td>4840</td>
      <td>고무제품 제조업</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DSR</td>
      <td>155660</td>
      <td>1차 비철금속 제조업</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GS</td>
      <td>78930</td>
      <td>기타 금융업</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GS글로벌</td>
      <td>1250</td>
      <td>상품 종합 도매업</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HDC현대산업개발</td>
      <td>294870</td>
      <td>건물 건설업</td>
    </tr>
  </tbody>
</table>
</div>




## 인덱싱

``pd객체.str[]``


```python
df['업종'].str[:-1].head()
```




    0       고무제품 제조
    1    1차 비철금속 제조
    2         기타 금융
    3      상품 종합 도매
    4         건물 건설
    Name: 업종, dtype: object



## 분할

``객체.str.split(" ", expand=True)``

expand=True는 결과를 DataFrame 객체로 만들기 위한 인수


```python
df['업종'].str.split(" ").head(3)
```




    0        [고무제품, 제조업]
    1    [1차, 비철금속, 제조업]
    2          [기타, 금융업]
    Name: 업종, dtype: object




```python
df1=df['업종'].str.split(" ", expand=True)
df1.tail()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2473</th>
      <td>자연과학</td>
      <td>및</td>
      <td>공학</td>
      <td>연구개발업</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2474</th>
      <td>소프트웨어</td>
      <td>개발</td>
      <td>및</td>
      <td>공급업</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2475</th>
      <td>자연과학</td>
      <td>및</td>
      <td>공학</td>
      <td>연구개발업</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2476</th>
      <td>기타</td>
      <td>화학제품</td>
      <td>제조업</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2477</th>
      <td>의료용</td>
      <td>기기</td>
      <td>제조업</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## 특정한 글자로 시작 여부를 확인 

``객체.str.startswith(word)``

인수인 word로 시작하면 True, 아닌 경우 False

## 마지막 글자가 지정된 글자와의 일치 여부를 확인 

``객체.str.endswith(word)``

인수인 word로 시작하면 True, 아닌 경우 False


```python
df1[0].str.startswith("자연과학").tail()
```




    2473     True
    2474    False
    2475     True
    2476    False
    2477    False
    Name: 0, dtype: bool




```python
df1[2].str.endswith("학").tail()
```




    2473     True
    2474    False
    2475     True
    2476    False
    2477    False
    Name: 2, dtype: object



## 포함글자 인식 

``객체.str.contains(word)``

객체에 word가 포함된 경우 True, 아닌 경우 False


```python
df['업종'].str.contains('금융').head()
```




    0    False
    1    False
    2     True
    3    False
    4    False
    Name: 업종, dtype: bool



## 문자위치 
 - ``객체.str.find(word)``: 왼쪽부터 검색후 위치반환 없으면 -1 반환
 - ``객체.str.rfind(sub=word)``: 오른쪽부터 검색후 위치반환 없으면 -1 반환


```python
x="Return highest indexes in each strings in the Series/Index."
x=pd.Series(str.split(x, " "))
x
```




    0           Return
    1          highest
    2          indexes
    3               in
    4             each
    5          strings
    6               in
    7              the
    8    Series/Index.
    dtype: object




```python
x.str.find('h')
```




    0   -1
    1    0
    2   -1
    3   -1
    4    3
    5   -1
    6   -1
    7    1
    8   -1
    dtype: int64




```python
x.str.rfind(sub='h')
```




    0   -1
    1    3
    2   -1
    3   -1
    4    3
    5   -1
    6   -1
    7    1
    8   -1
    dtype: int64



## 정규식에 부합하는 모든 값을 반환 

``객체.str.findall(정규식)``


```python
x.str.findall('\w+n')
```




    0    [Return]
    1          []
    2        [in]
    3        [in]
    4          []
    5     [strin]
    6        [in]
    7          []
    8        [In]
    dtype: object



## 객체중 a를 b로 대체

``객체.str.replace(a, b)''


```python
y=df['업종'].head()
y
```




    0       고무제품 제조업
    1    1차 비철금속 제조업
    2         기타 금융업
    3      상품 종합 도매업
    4         건물 건설업
    Name: 업종, dtype: object




```python
y.str.replace(' ', '_')
```




    0       고무제품_제조업
    1    1차_비철금속_제조업
    2         기타_금융업
    3      상품_종합_도매업
    4         건물_건설업
    Name: 업종, dtype: object



## 객체에서 지정한 문자(열)을 추출 

``객체.str.extract('(정규식)')``

위 식과 같이 지정한 문자(열)을 '( )'내에 입력하여 그룹화하여야 합니다. 없을 경우 NaN이 반환 


```python
y.str.extract('(\w+업)')
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
      <th>0</th>
      <td>제조업</td>
    </tr>
    <tr>
      <th>1</th>
      <td>제조업</td>
    </tr>
    <tr>
      <th>2</th>
      <td>금융업</td>
    </tr>
    <tr>
      <th>3</th>
      <td>도매업</td>
    </tr>
    <tr>
      <th>4</th>
      <td>건설업</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.str.extract('(금융업)')
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
      <th>0</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>금융업</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 객체 각 셀의 문자열 길이를 동일하게 하기 위해 지정된 기호, 문자, 숫자 등으로 채울 수 있습니다. 

``객체.str.pad(width, side,fillchar='')``

    - width: 고정길이<br>
    - side: 첨가할 방향 기본은 왼쪽부터<br>
    - fillchar: 채우기 위한 문자등, 기본값은 공백<br>


```python
y1=y.str.pad(width=15, side='left', fillchar="0")
y1
```




    0    0000000고무제품 제조업
    1    00001차 비철금속 제조업
    2    000000000기타 금융업
    3    000000상품 종합 도매업
    4    000000000건물 건설업
    Name: 업종, dtype: object




```python
for i in y1:
    print(len(i))
```

    15
    15
    15
    15
    15


## 문자(열)의 전과 후의 공백제거, 문자 사이의 공백은 영향 없음 

``객체.str.strip()``: 왼쪽, 오른쪽(전후)의 모든 공백제거

``객체.str.lstrip()``: 왼쪽 공백 제거

``객체.str.rstrip()``: 오른쪽 공백 제거 



```python
y=pd.Series(['   a b c   ', 'd   ef ', 'g h    i '])
for i in y:
    print(f'{i}, 길이: {len(i)}')
```

       a b c   , 길이: 11
    d   ef , 길이: 7
    g h    i , 길이: 9



```python
y1=y.str.strip()
for i in y1:
    print(f'{i}, 길이: {len(i)}')
```

    a b c, 길이: 5
    d   ef, 길이: 6
    g h    i, 길이: 8



```python
y1=y.str.lstrip()
for i in y1:
    print(f'{i}, 길이: {len(i)}')
```

    a b c   , 길이: 8
    d   ef , 길이: 7
    g h    i , 길이: 9



```python
y1=y.str.rstrip()
for i in y1:
    print(f'{i}, 길이: {len(i)}')
```

       a b c, 길이: 8
    d   ef, 길이: 6
    g h    i, 길이: 8


## 대소문자 변환 

``객체.str.lower()``: 소문자로 변환 

``객체.str.upper()``: 대문자로 변환 

``객체.str.swapcase()``: 대문자 $\rightarrow$ 소문자, 소문자 $\rightarrow$ 대문자


```python
y=pd.Series(['ABC', 'def', 'gHI'])
y
```




    0    ABC
    1    def
    2    gHI
    dtype: object




```python
y.str.lower()
```




    0    abc
    1    def
    2    ghi
    dtype: object




```python
y.str.upper()
```




    0    ABC
    1    DEF
    2    GHI
    dtype: object




```python
y.str.swapcase()
```




    0    abc
    1    DEF
    2    Ghi
    dtype: object


