```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv('home_data.csv')
```


```python
##NUMPY VERSION

np.__version__
```




    '1.21.5'




```python
## Number of records in the dataset


df.shape[0]
```




    11914




```python
##The most popular car manufacturers


df['Make'].value_counts()
```




    Chevrolet        1123
    Ford              881
    Volkswagen        809
    Toyota            746
    Dodge             626
    Nissan            558
    GMC               515
    Honda             449
    Mazda             423
    Cadillac          397
    Mercedes-Benz     353
    Suzuki            351
    BMW               334
    Infiniti          330
    Audi              328
    Hyundai           303
    Volvo             281
    Subaru            256
    Acura             252
    Kia               231
    Mitsubishi        213
    Lexus             202
    Buick             196
    Chrysler          187
    Pontiac           186
    Lincoln           164
    Oldsmobile        150
    Land Rover        143
    Porsche           136
    Saab              111
    Aston Martin       93
    Plymouth           82
    Bentley            74
    Ferrari            69
    FIAT               62
    Scion              60
    Maserati           58
    Lamborghini        52
    Rolls-Royce        31
    Lotus              29
    Tesla              18
    HUMMER             17
    Maybach            16
    Alfa Romeo          5
    McLaren             5
    Spyker              3
    Genesis             3
    Bugatti             3
    Name: Make, dtype: int64




```python
##Number of unique Audi car models

mk = df[df['Make'] == 'Audi']
```


```python
mk.Model.nunique()
```




    34




```python
##Number of columns with missing values

df.isnull().sum()
```




    Make                    0
    Model                   0
    Year                    0
    Engine Fuel Type        3
    Engine HP              69
    Engine Cylinders       30
    Transmission Type       0
    Driven_Wheels           0
    Number of Doors         6
    Market Category      3742
    Vehicle Size            0
    Vehicle Style           0
    highway MPG             0
    city mpg                0
    Popularity              0
    MSRP                    0
    dtype: int64




```python
##find the median value of "Engine Cylinders" column in the dataset

df['Engine Cylinders'].median()
```




    6.0




```python
## calculate the most frequent value of the same "Engine Cylinders"

df['Engine Cylinders'].mode()
```




    0    4.0
    Name: Engine Cylinders, dtype: float64




```python
##Use the fillna method to fill the missing values in "Engine Cylinders" with the most frequent value from the previous step

df['Engine Cylinders'].fillna(4.0)
```




    0        6.0
    1        6.0
    2        6.0
    3        6.0
    4        6.0
            ... 
    11909    6.0
    11910    6.0
    11911    6.0
    11912    6.0
    11913    6.0
    Name: Engine Cylinders, Length: 11914, dtype: float64




```python
## Now, calculate the median value of "Engine Cylinders" once again

df['Engine Cylinders'].median()
```




    6.0




```python
##Select all the "Lotus" cars from the dataset.

lotus = df[df['Make']=='Lotus']
```


```python
## Select only columns "Engine HP", "Engine Cylinders"

lotus1 = lotus[['Engine HP', 'Engine Cylinders']]
```


```python
##Now drop all duplicated rows using drop_duplicates method 

lotus1 = lotus1.drop_duplicates()
```


```python
##Get the underlying NumPy array. Let's call it X

X = np.array(lotus1)
```


```python
X
```




    array([[189.,   4.],
           [218.,   4.],
           [217.,   4.],
           [350.,   8.],
           [400.,   6.],
           [276.,   6.],
           [345.,   6.],
           [257.,   4.],
           [240.,   4.]])




```python
##transposing X as TX

TX = X.transpose()
```


```python
TX
```




    array([[189., 218., 217., 350., 400., 276., 345., 257., 240.],
           [  4.,   4.,   4.,   8.,   6.,   6.,   6.,   4.,   4.]])




```python
## matrix-matrix multiplication of XTX and X

XTX = TX.dot(X)
```


```python
XTX
```




    array([[7.31684e+05, 1.34100e+04],
           [1.34100e+04, 2.52000e+02]])




```python
## inverting XTX as inv_XTX

inv_XTX = np.linalg.inv(XTX)
```


```python
inv_XTX
```




    array([[ 5.53084235e-05, -2.94319825e-03],
           [-2.94319825e-03,  1.60588447e-01]])




```python
##crating an array y

y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])
```


```python
y
```




    array([1100,  800,  750,  850, 1300, 1000, 1000, 1300,  800])




```python
## Multiply the inverse of XTX with the transpose of X as XTX_TX

XTX_TX = inv_XTX.dot(TX)
```


```python
XTX_TX
```




    array([[-1.31950096e-03,  2.84443321e-04,  2.29134897e-04,
            -4.18763778e-03,  4.46417989e-03, -2.39406462e-03,
             1.42221660e-03,  2.44147184e-03,  1.50122864e-03],
           [ 8.60893170e-02,  7.36567735e-04,  3.67976598e-03,
             2.54588185e-01, -2.13748621e-01,  1.51207962e-01,
            -5.18727169e-02, -1.14048164e-01, -6.40137937e-02]])




```python
## MULTIPLYING XTX_TX BY Y AS W
W = XTX_TX.dot(y)
```


```python
W
```




    array([  4.59494481, -63.56432501])




```python
W[0]
```




    4.594944810094551




```python

```
