 |표현식|<center>의 미</center>|
 |:---:|:---|
 |^x| x 문자로 시작|
|x$| x 문자로 종료|
 |.x| .은 임의의 한 문자의 자리수를 표현<br> x로 끝남|
| x+| 반복을 표현<br> x 문자가 한번 이상 반복|
 |x?| ?는 존재여부를 표현<br> x 문자가 존재/비존재를 의미|
 |x*| *는 반복여부를 표현<br>x 문자가 0번 또는 그 이상 반복|
| x\|y| or 를 표현<br> x 또는 y 문자가 존재|
 |(x)| '()'는 그룹을 표현<br> x 를 그룹으로 처리|
 |(x)(y)|그룹들의 집합<br>앞에서부터 순서대로 번호를 부여하여 관리<br>x, y 는 각 그룹의 데이터로 관리됨|
 |(x)(?:y)|그룹들의 집합에 대한 예외를 표현<br>그룹 집합으로 관리되지 않음| 
 |x{n}| x 문자가 n번 반복|
 |x{n,}| x 문자가 n번 이상 반복|
 |x{n,m}| x 문자가 최소 n번 이상 최대 m 번 이하로 반복|

정규식에서 '[]'는 내부에 지정된 문자열의 범위 중에서 한 문자만을 선택함을 나타냄

|표현식|<center>의미</center>|
 |:---:|:---|
|[xy] |선택을 문자들 중 x 와 y 중에 하나를 의미|
|[\^xy]| 대괄호내의 ^은 not을 표현,  x 및 y 를 제외한 문자를 의미|
| [x-z]|range를 표현, x ~ z 사이의 문자를 의미 |
|\^| escape 를 표현하며 ^를 문자로 사용함을 의미|
|\b| word boundary를 표현하며 문자와 공백사이의 문자를 의미|
|\B| non word boundary를 표현하며 문자와 공백사이가 아닌 문자를 의미|
|\d| digit 를 표현하며 숫자를 의미 |
| \D| non digit 를 표현하며 숫자가 아닌 것을 의미| 
|\s| space 를 표현하며 공백 문자를 의미. |
| \S| non space를 표현하며 공백 문자가 아닌 것을 의미|
|\t| tab 을 표현하며 탭 문자를 의미|
|\v| vertical tab을 표현하며 수직 탭(?) 문자를 의미|
|\w| word 를 표현하며 알파벳 + 숫자 + _ 중의 한 문자임을 의미 |
| \W| non word를 표현하며 알파벳 + 숫자 + _ 가 아닌 문자를 의미| 

정규표현식에서 Flag를 사용할 수 있습니다. 이를 사용하지 않으면 문자열에 대해서 검색을 한번만 처리하고 종료합니다. 

<table border="1", cellpadding="20">
    <tbody>
        <tr>
        <th><center>Flag</center></th> 
        <th><center>의미</center></th>
        </tr>
        <tr>
        <td>g</td>
        <td> <center>Global 의 표현</center><br/> <center>대상 문자열내에 모든 패턴들을 검색하는 것을 의미 </center></td>
        </tr>        
        <tr>
        <td>i</td>
        <td> <center>Ignore case 를 표현</center><br/><center> 대상 문자열에 대해서 대/소문자를 식별하지 않는 것을 의미</center> </td>
        </tr>
        <tr>
        <td>m</td>
        <td> <center>Multi line을 표현</center><br/> <center>대상 문자열이 다중 라인의 문자열인 경우에도 검색하는 것을 의미</center></td>
        </tr>
</tbody>
</table>

예) 
- /[0-9]/g : 0~9사이의 임의의 숫자 한개를 찾음
- /[to]/g: 전체엣 t 또는 o를 모두 찾음
- /filter/g: 전체에서 filter 단어와 매칭되는 것을 찾음
