# Parkison's Disease Detection

### importing dependencies 


```python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as ac
from sklearn.preprocessing import StandardScaler
from sklearn import svm

```

### data collection and analysis


#### loading data from csv file to a Pandas DataFrame




```python
parkinsons_data = pd.read_csv('parkinsons.csv')
```


### printing head (first 5 rows) of dataframe




```python

parkinsons_data.head()

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
      <th>name</th>
      <th>MDVP:Fo(Hz)</th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>...</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>status</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>phon_R01_S01_1</td>
      <td>119.992</td>
      <td>157.302</td>
      <td>74.997</td>
      <td>0.00784</td>
      <td>0.00007</td>
      <td>0.00370</td>
      <td>0.00554</td>
      <td>0.01109</td>
      <td>0.04374</td>
      <td>...</td>
      <td>0.06545</td>
      <td>0.02211</td>
      <td>21.033</td>
      <td>1</td>
      <td>0.414783</td>
      <td>0.815285</td>
      <td>-4.813031</td>
      <td>0.266482</td>
      <td>2.301442</td>
      <td>0.284654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phon_R01_S01_2</td>
      <td>122.400</td>
      <td>148.650</td>
      <td>113.819</td>
      <td>0.00968</td>
      <td>0.00008</td>
      <td>0.00465</td>
      <td>0.00696</td>
      <td>0.01394</td>
      <td>0.06134</td>
      <td>...</td>
      <td>0.09403</td>
      <td>0.01929</td>
      <td>19.085</td>
      <td>1</td>
      <td>0.458359</td>
      <td>0.819521</td>
      <td>-4.075192</td>
      <td>0.335590</td>
      <td>2.486855</td>
      <td>0.368674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phon_R01_S01_3</td>
      <td>116.682</td>
      <td>131.111</td>
      <td>111.555</td>
      <td>0.01050</td>
      <td>0.00009</td>
      <td>0.00544</td>
      <td>0.00781</td>
      <td>0.01633</td>
      <td>0.05233</td>
      <td>...</td>
      <td>0.08270</td>
      <td>0.01309</td>
      <td>20.651</td>
      <td>1</td>
      <td>0.429895</td>
      <td>0.825288</td>
      <td>-4.443179</td>
      <td>0.311173</td>
      <td>2.342259</td>
      <td>0.332634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>phon_R01_S01_4</td>
      <td>116.676</td>
      <td>137.871</td>
      <td>111.366</td>
      <td>0.00997</td>
      <td>0.00009</td>
      <td>0.00502</td>
      <td>0.00698</td>
      <td>0.01505</td>
      <td>0.05492</td>
      <td>...</td>
      <td>0.08771</td>
      <td>0.01353</td>
      <td>20.644</td>
      <td>1</td>
      <td>0.434969</td>
      <td>0.819235</td>
      <td>-4.117501</td>
      <td>0.334147</td>
      <td>2.405554</td>
      <td>0.368975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>phon_R01_S01_5</td>
      <td>116.014</td>
      <td>141.781</td>
      <td>110.655</td>
      <td>0.01284</td>
      <td>0.00011</td>
      <td>0.00655</td>
      <td>0.00908</td>
      <td>0.01966</td>
      <td>0.06425</td>
      <td>...</td>
      <td>0.10470</td>
      <td>0.01767</td>
      <td>19.649</td>
      <td>1</td>
      <td>0.417356</td>
      <td>0.823484</td>
      <td>-3.747787</td>
      <td>0.234513</td>
      <td>2.332180</td>
      <td>0.410335</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



### rows and columns


```python

parkinsons_data.shape 
```




    (195, 24)



### info check


```python

parkinsons_data.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 195 entries, 0 to 194
    Data columns (total 24 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              195 non-null    object 
     1   MDVP:Fo(Hz)       195 non-null    float64
     2   MDVP:Fhi(Hz)      195 non-null    float64
     3   MDVP:Flo(Hz)      195 non-null    float64
     4   MDVP:Jitter(%)    195 non-null    float64
     5   MDVP:Jitter(Abs)  195 non-null    float64
     6   MDVP:RAP          195 non-null    float64
     7   MDVP:PPQ          195 non-null    float64
     8   Jitter:DDP        195 non-null    float64
     9   MDVP:Shimmer      195 non-null    float64
     10  MDVP:Shimmer(dB)  195 non-null    float64
     11  Shimmer:APQ3      195 non-null    float64
     12  Shimmer:APQ5      195 non-null    float64
     13  MDVP:APQ          195 non-null    float64
     14  Shimmer:DDA       195 non-null    float64
     15  NHR               195 non-null    float64
     16  HNR               195 non-null    float64
     17  status            195 non-null    int64  
     18  RPDE              195 non-null    float64
     19  DFA               195 non-null    float64
     20  spread1           195 non-null    float64
     21  spread2           195 non-null    float64
     22  D2                195 non-null    float64
     23  PPE               195 non-null    float64
    dtypes: float64(22), int64(1), object(1)
    memory usage: 36.7+ KB
    

### checking for missing values


```python
parkinsons_data.isnull().sum()
```




    name                0
    MDVP:Fo(Hz)         0
    MDVP:Fhi(Hz)        0
    MDVP:Flo(Hz)        0
    MDVP:Jitter(%)      0
    MDVP:Jitter(Abs)    0
    MDVP:RAP            0
    MDVP:PPQ            0
    Jitter:DDP          0
    MDVP:Shimmer        0
    MDVP:Shimmer(dB)    0
    Shimmer:APQ3        0
    Shimmer:APQ5        0
    MDVP:APQ            0
    Shimmer:DDA         0
    NHR                 0
    HNR                 0
    status              0
    RPDE                0
    DFA                 0
    spread1             0
    spread2             0
    D2                  0
    PPE                 0
    dtype: int64



### stats on the given data


```python
parkinsons_data.describe()
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
      <th>MDVP:Fo(Hz)</th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>MDVP:Shimmer(dB)</th>
      <th>...</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>status</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>...</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>154.228641</td>
      <td>197.104918</td>
      <td>116.324631</td>
      <td>0.006220</td>
      <td>0.000044</td>
      <td>0.003306</td>
      <td>0.003446</td>
      <td>0.009920</td>
      <td>0.029709</td>
      <td>0.282251</td>
      <td>...</td>
      <td>0.046993</td>
      <td>0.024847</td>
      <td>21.885974</td>
      <td>0.753846</td>
      <td>0.498536</td>
      <td>0.718099</td>
      <td>-5.684397</td>
      <td>0.226510</td>
      <td>2.381826</td>
      <td>0.206552</td>
    </tr>
    <tr>
      <th>std</th>
      <td>41.390065</td>
      <td>91.491548</td>
      <td>43.521413</td>
      <td>0.004848</td>
      <td>0.000035</td>
      <td>0.002968</td>
      <td>0.002759</td>
      <td>0.008903</td>
      <td>0.018857</td>
      <td>0.194877</td>
      <td>...</td>
      <td>0.030459</td>
      <td>0.040418</td>
      <td>4.425764</td>
      <td>0.431878</td>
      <td>0.103942</td>
      <td>0.055336</td>
      <td>1.090208</td>
      <td>0.083406</td>
      <td>0.382799</td>
      <td>0.090119</td>
    </tr>
    <tr>
      <th>min</th>
      <td>88.333000</td>
      <td>102.145000</td>
      <td>65.476000</td>
      <td>0.001680</td>
      <td>0.000007</td>
      <td>0.000680</td>
      <td>0.000920</td>
      <td>0.002040</td>
      <td>0.009540</td>
      <td>0.085000</td>
      <td>...</td>
      <td>0.013640</td>
      <td>0.000650</td>
      <td>8.441000</td>
      <td>0.000000</td>
      <td>0.256570</td>
      <td>0.574282</td>
      <td>-7.964984</td>
      <td>0.006274</td>
      <td>1.423287</td>
      <td>0.044539</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>117.572000</td>
      <td>134.862500</td>
      <td>84.291000</td>
      <td>0.003460</td>
      <td>0.000020</td>
      <td>0.001660</td>
      <td>0.001860</td>
      <td>0.004985</td>
      <td>0.016505</td>
      <td>0.148500</td>
      <td>...</td>
      <td>0.024735</td>
      <td>0.005925</td>
      <td>19.198000</td>
      <td>1.000000</td>
      <td>0.421306</td>
      <td>0.674758</td>
      <td>-6.450096</td>
      <td>0.174351</td>
      <td>2.099125</td>
      <td>0.137451</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>148.790000</td>
      <td>175.829000</td>
      <td>104.315000</td>
      <td>0.004940</td>
      <td>0.000030</td>
      <td>0.002500</td>
      <td>0.002690</td>
      <td>0.007490</td>
      <td>0.022970</td>
      <td>0.221000</td>
      <td>...</td>
      <td>0.038360</td>
      <td>0.011660</td>
      <td>22.085000</td>
      <td>1.000000</td>
      <td>0.495954</td>
      <td>0.722254</td>
      <td>-5.720868</td>
      <td>0.218885</td>
      <td>2.361532</td>
      <td>0.194052</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>182.769000</td>
      <td>224.205500</td>
      <td>140.018500</td>
      <td>0.007365</td>
      <td>0.000060</td>
      <td>0.003835</td>
      <td>0.003955</td>
      <td>0.011505</td>
      <td>0.037885</td>
      <td>0.350000</td>
      <td>...</td>
      <td>0.060795</td>
      <td>0.025640</td>
      <td>25.075500</td>
      <td>1.000000</td>
      <td>0.587562</td>
      <td>0.761881</td>
      <td>-5.046192</td>
      <td>0.279234</td>
      <td>2.636456</td>
      <td>0.252980</td>
    </tr>
    <tr>
      <th>max</th>
      <td>260.105000</td>
      <td>592.030000</td>
      <td>239.170000</td>
      <td>0.033160</td>
      <td>0.000260</td>
      <td>0.021440</td>
      <td>0.019580</td>
      <td>0.064330</td>
      <td>0.119080</td>
      <td>1.302000</td>
      <td>...</td>
      <td>0.169420</td>
      <td>0.314820</td>
      <td>33.047000</td>
      <td>1.000000</td>
      <td>0.685151</td>
      <td>0.825288</td>
      <td>-2.434031</td>
      <td>0.450493</td>
      <td>3.671155</td>
      <td>0.527367</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
</div>



### target variable distribution 


```python
parkinsons_data['status'].value_counts()
```




    status
    1    147
    0     48
    Name: count, dtype: int64


0  --> Healthy
1  --> Parkinson's Positive 
### data grouping based on target varible


```python
#parkinsons_data.groupby('status').mean()
```
Data Pre-ProcessingSeparating the Features & Target

```python
x = parkinsons_data.drop(columns=['name', 'status'], axis = 1)
y = parkinsons_data['status']
```


```python
print(x)
```

         MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)  MDVP:Jitter(%)  \
    0        119.992       157.302        74.997         0.00784   
    1        122.400       148.650       113.819         0.00968   
    2        116.682       131.111       111.555         0.01050   
    3        116.676       137.871       111.366         0.00997   
    4        116.014       141.781       110.655         0.01284   
    ..           ...           ...           ...             ...   
    190      174.188       230.978        94.261         0.00459   
    191      209.516       253.017        89.488         0.00564   
    192      174.688       240.005        74.287         0.01360   
    193      198.764       396.961        74.904         0.00740   
    194      214.289       260.277        77.973         0.00567   
    
         MDVP:Jitter(Abs)  MDVP:RAP  MDVP:PPQ  Jitter:DDP  MDVP:Shimmer  \
    0             0.00007   0.00370   0.00554     0.01109       0.04374   
    1             0.00008   0.00465   0.00696     0.01394       0.06134   
    2             0.00009   0.00544   0.00781     0.01633       0.05233   
    3             0.00009   0.00502   0.00698     0.01505       0.05492   
    4             0.00011   0.00655   0.00908     0.01966       0.06425   
    ..                ...       ...       ...         ...           ...   
    190           0.00003   0.00263   0.00259     0.00790       0.04087   
    191           0.00003   0.00331   0.00292     0.00994       0.02751   
    192           0.00008   0.00624   0.00564     0.01873       0.02308   
    193           0.00004   0.00370   0.00390     0.01109       0.02296   
    194           0.00003   0.00295   0.00317     0.00885       0.01884   
    
         MDVP:Shimmer(dB)  ...  MDVP:APQ  Shimmer:DDA      NHR     HNR      RPDE  \
    0               0.426  ...   0.02971      0.06545  0.02211  21.033  0.414783   
    1               0.626  ...   0.04368      0.09403  0.01929  19.085  0.458359   
    2               0.482  ...   0.03590      0.08270  0.01309  20.651  0.429895   
    3               0.517  ...   0.03772      0.08771  0.01353  20.644  0.434969   
    4               0.584  ...   0.04465      0.10470  0.01767  19.649  0.417356   
    ..                ...  ...       ...          ...      ...     ...       ...   
    190             0.405  ...   0.02745      0.07008  0.02764  19.517  0.448439   
    191             0.263  ...   0.01879      0.04812  0.01810  19.147  0.431674   
    192             0.256  ...   0.01667      0.03804  0.10715  17.883  0.407567   
    193             0.241  ...   0.01588      0.03794  0.07223  19.020  0.451221   
    194             0.190  ...   0.01373      0.03078  0.04398  21.209  0.462803   
    
              DFA   spread1   spread2        D2       PPE  
    0    0.815285 -4.813031  0.266482  2.301442  0.284654  
    1    0.819521 -4.075192  0.335590  2.486855  0.368674  
    2    0.825288 -4.443179  0.311173  2.342259  0.332634  
    3    0.819235 -4.117501  0.334147  2.405554  0.368975  
    4    0.823484 -3.747787  0.234513  2.332180  0.410335  
    ..        ...       ...       ...       ...       ...  
    190  0.657899 -6.538586  0.121952  2.657476  0.133050  
    191  0.683244 -6.195325  0.129303  2.784312  0.168895  
    192  0.655683 -6.787197  0.158453  2.679772  0.131728  
    193  0.643956 -6.744577  0.207454  2.138608  0.123306  
    194  0.664357 -5.724056  0.190667  2.555477  0.148569  
    
    [195 rows x 22 columns]
    


```python
print(y)
```

    0      1
    1      1
    2      1
    3      1
    4      1
          ..
    190    0
    191    0
    192    0
    193    0
    194    0
    Name: status, Length: 195, dtype: int64
    

### splitting the data to training data & test data


```python
x_train, x_test, y_train, y_test = tts(x,y,test_size=.2,random_state = 2)
```


```python
print  (x.shape,x_train.shape, x_test.shape)
```

    (195, 22) (156, 22) (39, 22)
    

### data standardization


```python
scaler = StandardScaler()
```


```python
scaler.fit(x_train)
```




<style>#sk-container-id-8 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-8 {
  color: var(--sklearn-color-text);
}

#sk-container-id-8 pre {
  padding: 0;
}

#sk-container-id-8 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-8 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-8 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-8 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-8 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-8 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-8 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-8 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-8 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-8 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-8 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-8 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-8 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-8 div.sk-label label.sk-toggleable__label,
#sk-container-id-8 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-8 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-8 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-8 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-8 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-8 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-8 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-8 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-8 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-8 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>StandardScaler()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;StandardScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div>




```python
x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
```


```python
print(x_train)
```

    [[ 0.63239631 -0.02731081 -0.87985049 ... -0.97586547 -0.55160318
       0.07769494]
     [-1.05512719 -0.83337041 -0.9284778  ...  0.3981808  -0.61014073
       0.39291782]
     [ 0.02996187 -0.29531068 -1.12211107 ... -0.43937044 -0.62849605
      -0.50948408]
     ...
     [-0.9096785  -0.6637302  -0.160638   ...  1.22001022 -0.47404629
      -0.2159482 ]
     [-0.35977689  0.19731822 -0.79063679 ... -0.17896029 -0.47272835
       0.28181221]
     [ 1.01957066  0.19922317 -0.61914972 ... -0.716232    1.23632066
      -0.05829386]]
    
Model Training

Support Vector Machine Model

```python
model = svm.SVC(kernel = 'linear')

```

### training SVM model with training data


```python
model.fit(x_train, y_train)

```




<style>#sk-container-id-9 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-9 {
  color: var(--sklearn-color-text);
}

#sk-container-id-9 pre {
  padding: 0;
}

#sk-container-id-9 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-9 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-9 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-9 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-9 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-9 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-9 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-9 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-9 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-9 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-9 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-9 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-9 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-9 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-9 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-9 div.sk-label label.sk-toggleable__label,
#sk-container-id-9 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-9 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-9 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-9 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-9 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-9 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-9 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-9 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-9 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-9 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-9 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-9 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-9" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC(kernel=&#x27;linear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" checked><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;SVC<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>SVC(kernel=&#x27;linear&#x27;)</pre></div> </div></div></div></div>



## Model Evaluation 



### Accuracy Score

#### Accuracy score on training data


```python
x_train_prediction = model.predict(x_train)
training_data_accuracy = ac(y_train, x_train_prediction)
```


```python
print("Accuracy score of the training data: ",training_data_accuracy)
```

    Accuracy score of the training data:  0.8846153846153846
    

#### Accuracy of the testing data


```python
x_test_prediction = model.predict(x_test)
testing_data_accuracy = ac(y_test, x_test_prediction)
```


```python
print("Accuracy score of the training data: ",training_data_accuracy)
```

    Accuracy score of the training data:  0.8846153846153846
    

### Building a Predective System


```python
input_data = (198.45800,219.29000,148.69100,0.00376,0.00002,0.00182,0.00215,0.00546,0.03527,0.29700,0.02055,0.02076,0.02530,0.06165,0.01728,18.70200,0.606273,0.661735,-5.585259,0.310746,2.465528,0.209863)
```

### changing input data to numpy array


```python
input_data_as_numpy_array = np.asarray(input_data)
                                                        
```

#### reshape the numpy array


```python
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

```

### standardize the data


```python
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)

if prediction == [1]:
    print("This person has Parkinson's Disease")
else:
    print("NO")    


```

    This person has Parkinson's Disease
    

    C:\Users\lenovo\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
      warnings.warn(
    


```python

```


```python

```


```python

```
