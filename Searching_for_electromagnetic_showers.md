
<a href="https://colab.research.google.com/github/MohamedElashri/Hadron-Collider-ML/blob/master/Searching_for_electromagnetic_showers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
# Sample solution is based on nontebooks by Artem Borzov

import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import tables as tb
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from sklearn.neighbors import BallTree, KDTree, DistanceMetric
import glob

%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib


# Load Datasets


```python
!wget https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/4/training.tgz
!wget https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/4/test.h5.gz
```

    --2021-03-01 20:27:08--  https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/3/training.tgz
    Resolving github.com (github.com)... 140.82.112.4
    Connecting to github.com (github.com)|140.82.112.4|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://github-releases.githubusercontent.com/187530009/1cab5c80-7a8a-11eb-8fee-e637a5b696c3?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T202708Z&X-Amz-Expires=300&X-Amz-Signature=77c100e6bd7c2a309f59c49e04a09795c0cc11a18c96fe53695b0abdf64814d8&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dtraining.tgz&response-content-type=application%2Foctet-stream [following]
    --2021-03-01 20:27:08--  https://github-releases.githubusercontent.com/187530009/1cab5c80-7a8a-11eb-8fee-e637a5b696c3?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T202708Z&X-Amz-Expires=300&X-Amz-Signature=77c100e6bd7c2a309f59c49e04a09795c0cc11a18c96fe53695b0abdf64814d8&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dtraining.tgz&response-content-type=application%2Foctet-stream
    Resolving github-releases.githubusercontent.com (github-releases.githubusercontent.com)... 185.199.108.154, 185.199.109.154, 185.199.111.154, ...
    Connecting to github-releases.githubusercontent.com (github-releases.githubusercontent.com)|185.199.108.154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1352765586 (1.3G) [application/octet-stream]
    Saving to: ‘training.tgz’
    
    training.tgz        100%[===================>]   1.26G  53.6MB/s    in 22s     
    
    2021-03-01 20:27:30 (58.6 MB/s) - ‘training.tgz’ saved [1352765586/1352765586]
    
    --2021-03-01 20:27:30--  https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/3/test.h5.gz
    Resolving github.com (github.com)... 140.82.113.4
    Connecting to github.com (github.com)|140.82.113.4|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://github-releases.githubusercontent.com/187530009/04d3d880-7a8a-11eb-8875-7062affa6661?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T202730Z&X-Amz-Expires=300&X-Amz-Signature=5c63231ac85bacd941c4465101b8b3534296721ef7f3148a2b0084ded881e876&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dtest.h5.gz&response-content-type=application%2Foctet-stream [following]
    --2021-03-01 20:27:30--  https://github-releases.githubusercontent.com/187530009/04d3d880-7a8a-11eb-8875-7062affa6661?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T202730Z&X-Amz-Expires=300&X-Amz-Signature=5c63231ac85bacd941c4465101b8b3534296721ef7f3148a2b0084ded881e876&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dtest.h5.gz&response-content-type=application%2Foctet-stream
    Resolving github-releases.githubusercontent.com (github-releases.githubusercontent.com)... 185.199.111.154, 185.199.109.154, 185.199.110.154, ...
    Connecting to github-releases.githubusercontent.com (github-releases.githubusercontent.com)|185.199.111.154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 136352106 (130M) [application/octet-stream]
    Saving to: ‘test.h5.gz’
    
    test.h5.gz          100%[===================>] 130.04M  35.1MB/s    in 3.7s    
    
    2021-03-01 20:27:34 (35.1 MB/s) - ‘test.h5.gz’ saved [136352106/136352106]
    



```python
!gunzip -k *.gz
!gunzip -k *.tgz
!tar xf training.tar
```


```python
!wget https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/4/open30.h5
```

    --2021-03-01 20:34:46--  https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/3/open30.h5
    Resolving github.com (github.com)... 140.82.114.3
    Connecting to github.com (github.com)|140.82.114.3|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://github-releases.githubusercontent.com/187530009/884df380-7aa3-11eb-9408-b618d5e7d50e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T203446Z&X-Amz-Expires=300&X-Amz-Signature=8eb50353e4080da4f217faad8cf052d6bde5044849282be9fb961e69c976ed3c&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dopen30.h5&response-content-type=application%2Foctet-stream [following]
    --2021-03-01 20:34:46--  https://github-releases.githubusercontent.com/187530009/884df380-7aa3-11eb-9408-b618d5e7d50e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T203446Z&X-Amz-Expires=300&X-Amz-Signature=8eb50353e4080da4f217faad8cf052d6bde5044849282be9fb961e69c976ed3c&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dopen30.h5&response-content-type=application%2Foctet-stream
    Resolving github-releases.githubusercontent.com (github-releases.githubusercontent.com)... 185.199.111.154, 185.199.110.154, 185.199.109.154, ...
    Connecting to github-releases.githubusercontent.com (github-releases.githubusercontent.com)|185.199.111.154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 310377675 (296M) [application/octet-stream]
    Saving to: ‘open30.h5’
    
    open30.h5           100%[===================>] 296.00M  95.7MB/s    in 3.1s    
    
    2021-03-01 20:34:50 (95.7 MB/s) - ‘open30.h5’ saved [310377675/310377675]
    



```python
train = pd.read_hdf('open30.h5') # pick just a single brick
test = pd.read_hdf('test.h5')
```


```python
test = test.reset_index(drop=True)
```


```python
test.head()
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
      <th>TX</th>
      <th>TY</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>data_ind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.087240</td>
      <td>0.119438</td>
      <td>86536.218750</td>
      <td>62988.312500</td>
      <td>56892.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.380208</td>
      <td>0.198382</td>
      <td>93346.765625</td>
      <td>58062.937500</td>
      <td>14223.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.348549</td>
      <td>-0.099981</td>
      <td>66129.578125</td>
      <td>23038.673828</td>
      <td>46548.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.585342</td>
      <td>-0.126879</td>
      <td>68825.523438</td>
      <td>55186.625000</td>
      <td>45255.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.038579</td>
      <td>-0.131510</td>
      <td>36366.941406</td>
      <td>47564.878906</td>
      <td>9051.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.head()
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
      <th>TX</th>
      <th>TY</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>event_id</th>
      <th>signal</th>
      <th>data_ind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.248382</td>
      <td>0.072824</td>
      <td>37014.070312</td>
      <td>38318.898438</td>
      <td>31032.0</td>
      <td>-999.0</td>
      <td>0.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.768548</td>
      <td>0.490205</td>
      <td>46626.425781</td>
      <td>15151.117188</td>
      <td>72408.0</td>
      <td>188918.0</td>
      <td>1.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.132050</td>
      <td>-0.049405</td>
      <td>54031.640625</td>
      <td>69132.320312</td>
      <td>60771.0</td>
      <td>-999.0</td>
      <td>0.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.260584</td>
      <td>0.207106</td>
      <td>56653.531250</td>
      <td>35025.765625</td>
      <td>43962.0</td>
      <td>-999.0</td>
      <td>0.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.267597</td>
      <td>0.111272</td>
      <td>42231.714844</td>
      <td>47199.027344</td>
      <td>20688.0</td>
      <td>-999.0</td>
      <td>0.0</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



Let's see how showers look like inside one data_ind


```python
CMAP = sns.diverging_palette(220, 20, s=99, as_cmap=True, n=2500)

def plot3D(X, target, elev=0, azim=0, title=None, sub=111):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig)
    mappab = ax.scatter(x, y, z, c=target, cmap=CMAP)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

# Этот инструмент позволяет менять точку зрения
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(mappable=mappab, label='Target variable')
    plt.show()
    
feat_XY = ['TX', 'TY', 'X', 'Y']
```


```python
first = train.loc[train.data_ind == 21, :]
```


```python
plot3D(first.loc[first.signal==1, ['Z', 'X', 'Y']].values,
       first.loc[first.signal==1].signal.values, elev=20, azim=60)
```


![png](Searching_for_electromagnetic_showers_files/Searching_for_electromagnetic_showers_13_0.png)



```python
plot3D(first.loc[first.signal==1, ['Z', 'X', 'Y']].values,
       first.loc[first.signal==1].signal.values, elev=45, azim=0)
```


![png](Searching_for_electromagnetic_showers_files/Searching_for_electromagnetic_showers_14_0.png)


## Let's take a look at basetrack distribution along the axis


```python
axis = 'X'

fig = plt.figure(figsize = [20, 10])
fig.add_subplot(221)
plt.hist(first.loc[first.signal == 1, axis], bins=500, histtype='step')
fig.add_subplot(222)
plt.hist(first.loc[first.signal == 0, axis], bins=500, histtype='step')
fig.add_subplot(223)
values_X = plt.hist(first.loc[:, axis], bins=500, histtype='step')
plt.show()
```


![png](Searching_for_electromagnetic_showers_files/Searching_for_electromagnetic_showers_16_0.png)



```python
axis = 'Y'

fig = plt.figure(figsize = [20, 10])
fig.add_subplot(221)
plt.hist(first.loc[first.signal == 1, axis], bins=500, histtype='step')
fig.add_subplot(222)
plt.hist(first.loc[first.signal == 0, axis], bins=500, histtype='step')
fig.add_subplot(223)
values_X = plt.hist(first.loc[:, axis], bins=500, histtype='step')
plt.show()
```


![png](Searching_for_electromagnetic_showers_files/Searching_for_electromagnetic_showers_17_0.png)



```python
axis = 'Z'

fig = plt.figure(figsize = [20, 10])
fig.add_subplot(221)
plt.hist(first.loc[first.signal == 1, axis], bins=500, histtype='step')
fig.add_subplot(222)
plt.hist(first.loc[first.signal == 0, axis], bins=500, histtype='step')
fig.add_subplot(223)
values_X = plt.hist(first.loc[:, axis], bins=500, histtype='step')
plt.show()
```


![png](Searching_for_electromagnetic_showers_files/Searching_for_electromagnetic_showers_18_0.png)


# Let's group close tracks from neighbour plates into pairs


```python
def add_neighbours(df, k, metric='minkowski'):
    res = []
    
    for data_ind in tqdm(np.unique(df.data_ind)):
        ind = df.loc[df.data_ind == data_ind].copy()
        #как будет замечено, 1293 - это расстояние между слайсами по Z
        ind[['TX', 'TY']] *= 1293
        values = np.unique(ind.Z)
        
        for j in range(1, len(values)):
            z, z_next = (ind.loc[ind.Z == values[j-1]].copy(),
                         ind.loc[ind.Z == values[j]].copy())
            
            b_tree = BallTree(z_next[feat_XY], metric=metric)
            d, i = b_tree.query(z[feat_XY], k=min(k, len(z_next)))
            
            for m in range(i.shape[1]):
                data = z_next.iloc[i[:, m]]
                z_copy = z.copy()
                for col in feat_XY + ['Z']:
                    z_copy[col + '_pair'] = data[col].values
                res.append(z_copy)
            
        res.append(z_next)
        
    res = pd.concat(res, sort=True)
    for col in feat_XY + ['Z']:
        res['d' + col] = res[col].values - res[col + '_pair'].values
    return res

def balance_train(df, k):
    data = add_neighbours(df, k=k)
    noise = data.event_id == -999
    signal, not_signal = data.loc[np.logical_not(noise)], data.loc[noise]
    noise_part = not_signal.sample(len(signal))
    return pd.concat([signal, noise_part], sort=True).reset_index(drop=True)
```


```python
train = []
for file in glob.glob('open*.h5')[:5]: # just 5 bricks
    train.append(balance_train(pd.read_hdf(file), k=3))
train = pd.concat(train, sort=True)
```

    100%|██████████| 10/10 [00:43<00:00,  4.32s/it]



```python
train.head()
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
      <th>TX</th>
      <th>TX_pair</th>
      <th>TY</th>
      <th>TY_pair</th>
      <th>X</th>
      <th>X_pair</th>
      <th>Y</th>
      <th>Y_pair</th>
      <th>Z</th>
      <th>Z_pair</th>
      <th>dTX</th>
      <th>dTY</th>
      <th>dX</th>
      <th>dY</th>
      <th>dZ</th>
      <th>data_ind</th>
      <th>event_id</th>
      <th>signal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-120.490211</td>
      <td>74.753609</td>
      <td>444.966217</td>
      <td>397.600922</td>
      <td>38500.425781</td>
      <td>40560.890625</td>
      <td>25600.734375</td>
      <td>47039.101562</td>
      <td>3879.000000</td>
      <td>4154.886719</td>
      <td>-195.243820</td>
      <td>47.365295</td>
      <td>-2060.464844</td>
      <td>-21438.367188</td>
      <td>-275.886719</td>
      <td>21</td>
      <td>193609.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>64.776703</td>
      <td>74.753609</td>
      <td>363.636444</td>
      <td>397.600922</td>
      <td>40566.082031</td>
      <td>40560.890625</td>
      <td>47068.250000</td>
      <td>47039.101562</td>
      <td>3879.000000</td>
      <td>4154.886719</td>
      <td>-9.976906</td>
      <td>-33.964478</td>
      <td>5.191406</td>
      <td>29.148438</td>
      <td>-275.886719</td>
      <td>21</td>
      <td>183053.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>74.753609</td>
      <td>-106.954178</td>
      <td>397.600922</td>
      <td>454.387695</td>
      <td>40560.890625</td>
      <td>38509.781250</td>
      <td>47039.101562</td>
      <td>25566.177734</td>
      <td>4154.886719</td>
      <td>4158.167969</td>
      <td>181.707794</td>
      <td>-56.786774</td>
      <td>2051.109375</td>
      <td>21472.923828</td>
      <td>-3.281250</td>
      <td>21</td>
      <td>183053.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-106.954178</td>
      <td>-135.986267</td>
      <td>454.387695</td>
      <td>363.800568</td>
      <td>38509.781250</td>
      <td>38328.984375</td>
      <td>25566.177734</td>
      <td>25683.556641</td>
      <td>4158.167969</td>
      <td>5172.000000</td>
      <td>29.032089</td>
      <td>90.587128</td>
      <td>180.796875</td>
      <td>-117.378906</td>
      <td>-1013.832031</td>
      <td>21</td>
      <td>195432.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-106.954178</td>
      <td>-117.631310</td>
      <td>454.387695</td>
      <td>444.702545</td>
      <td>38509.781250</td>
      <td>38386.769531</td>
      <td>25566.177734</td>
      <td>26033.638672</td>
      <td>4158.167969</td>
      <td>5172.000000</td>
      <td>10.677132</td>
      <td>9.685150</td>
      <td>123.011719</td>
      <td>-467.460938</td>
      <td>-1013.832031</td>
      <td>21</td>
      <td>195432.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# Building a model


```python
y_train = train.signal
X_train = train.drop(['event_id', 'signal', 'data_ind'], axis=1)
```


```python
# import os

import xgboost as xg
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
```


```python
param_grid = {
        'n_estimators':[10, 20], 
        'max_depth':[15],
}

class XGBClassifier_tmp(XGBClassifier):
    def predict(self, X):
        return XGBClassifier.predict_proba(self, X)[:, 1]

clf = GridSearchCV(XGBClassifier_tmp(learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, n_jobs=20), 
                   param_grid=param_grid, n_jobs=1,
                   scoring='roc_auc',
                   cv=StratifiedKFold(3, shuffle=True, random_state=0),
                   verbose=7)
```


```python
clf.fit(X_train, y_train)
```

    Fitting 3 folds for each of 2 candidates, totalling 6 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV] max_depth=15, n_estimators=10 ...................................
    [CV] ....... max_depth=15, n_estimators=10, score=0.932, total= 1.1min
    [CV] max_depth=15, n_estimators=10 ...................................


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  1.1min remaining:    0.0s


    [CV] ....... max_depth=15, n_estimators=10, score=0.933, total= 1.0min
    [CV] max_depth=15, n_estimators=10 ...................................


    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:  2.1min remaining:    0.0s


    [CV] ....... max_depth=15, n_estimators=10, score=0.933, total= 1.0min
    [CV] max_depth=15, n_estimators=20 ...................................


    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:  3.1min remaining:    0.0s


    [CV] ....... max_depth=15, n_estimators=20, score=0.939, total= 2.0min
    [CV] max_depth=15, n_estimators=20 ...................................


    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:  5.2min remaining:    0.0s


    [CV] ....... max_depth=15, n_estimators=20, score=0.940, total= 2.0min
    [CV] max_depth=15, n_estimators=20 ...................................


    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:  7.2min remaining:    0.0s


    [CV] ....... max_depth=15, n_estimators=20, score=0.939, total= 2.0min


    [Parallel(n_jobs=1)]: Done   6 out of   6 | elapsed:  9.2min remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   6 out of   6 | elapsed:  9.2min finished





    GridSearchCV(cv=StratifiedKFold(n_splits=3, random_state=0, shuffle=True),
                 error_score=nan,
                 estimator=XGBClassifier_tmp(base_score=0.5, booster='gbtree',
                                             colsample_bylevel=1,
                                             colsample_bynode=1,
                                             colsample_bytree=0.8, gamma=0,
                                             learning_rate=0.05, max_delta_step=0,
                                             max_depth=3, min_child_weight=1,
                                             missing=None, n_estimators=100,
                                             n_jobs=20, nthread=None,
                                             objective='binary:logistic',
                                             random_state=0, reg_alpha=0,
                                             reg_lambda=1, scale_pos_weight=1,
                                             seed=None, silent=None, subsample=0.8,
                                             verbosity=1),
                 iid='deprecated', n_jobs=1,
                 param_grid={'max_depth': [15], 'n_estimators': [10, 20]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='roc_auc', verbose=7)




```python
clf.best_estimator_
```




    XGBClassifier_tmp(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=0.8, gamma=0,
                      learning_rate=0.05, max_delta_step=0, max_depth=15,
                      min_child_weight=1, missing=None, n_estimators=20, n_jobs=20,
                      nthread=None, objective='binary:logistic', random_state=0,
                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                      silent=None, subsample=0.8, verbosity=1)




```python
xgb_class = clf.best_estimator_
```

# DNN


```python
from sklearn.preprocessing import Normalizer
transformer = Normalizer()
X_train_norm = transformer.fit_transform(X_train.fillna(0))
```


```python
X_train_norm[:5]
```




    array([[-1.4953952e-03,  9.2776160e-04,  5.5224434e-03,  4.9345964e-03,
             4.7782600e-01,  5.0339824e-01,  3.1772885e-01,  5.8379889e-01,
             4.8141986e-02,  5.1565997e-02, -2.4231570e-03,  5.8784726e-04,
            -2.5572279e-02, -2.6607001e-01, -3.4240100e-03],
           [ 7.3573901e-04,  8.4905751e-04,  4.1302121e-03,  4.5159832e-03,
             4.6075282e-01,  4.6069387e-01,  5.3460497e-01,  5.3427392e-01,
             4.4057995e-02,  4.7191538e-02, -1.1331850e-04, -3.8577127e-04,
             5.8964408e-05,  3.3107030e-04, -3.1335435e-03],
           [ 9.2757843e-04, -1.3271384e-03,  4.9336217e-03,  5.6382590e-03,
             5.0329888e-01,  4.7784773e-01,  5.8368361e-01,  3.1723732e-01,
             5.1555816e-02,  5.1596530e-02,  2.2547168e-03, -7.0463738e-04,
             2.5451142e-02,  2.6644626e-01, -4.0715313e-05],
           [-1.6289761e-03, -2.0711522e-03,  6.9205961e-03,  5.5408999e-03,
             5.8652699e-01,  5.8377337e-01,  3.8938817e-01,  3.9117593e-01,
             6.3331380e-02,  7.8772649e-02,  4.4217607e-04,  1.3796961e-03,
             2.7536445e-03, -1.7877509e-03, -1.5441268e-02],
           [-1.6246896e-03, -1.7868808e-03,  6.9023855e-03,  6.7552631e-03,
             5.8498359e-01,  5.8311498e-01,  3.8836354e-01,  3.9546451e-01,
             6.3164733e-02,  7.8565367e-02,  1.6219118e-04,  1.4712247e-04,
             1.8686120e-03, -7.1009747e-03, -1.5400636e-02]], dtype=float32)




```python
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
```


```python
def nn_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model
```


```python
callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'),
            ModelCheckpoint('output/{val_loss:.4f}.hdf5', monitor='val_loss', verbose=2, save_best_only=True, mode='auto')]
```


```python
nn = nn_model(X_train_norm.shape[1])
nn.fit(X_train_norm, y_train, validation_split=0.2, epochs=20, verbose=2, batch_size=256, shuffle=True, callbacks=callbacks)
```

    Epoch 1/20
    3085/3085 - 19s - loss: 0.5540 - val_loss: 0.7758
    
    Epoch 00001: val_loss improved from inf to 0.77578, saving model to output/0.7758.hdf5
    Epoch 2/20
    3085/3085 - 18s - loss: 0.5376 - val_loss: 0.8416
    
    Epoch 00002: val_loss did not improve from 0.77578
    Epoch 3/20
    3085/3085 - 18s - loss: 0.5315 - val_loss: 0.8182
    
    Epoch 00003: val_loss did not improve from 0.77578
    Epoch 4/20
    3085/3085 - 18s - loss: 0.5280 - val_loss: 0.7537
    
    Epoch 00004: val_loss improved from 0.77578 to 0.75375, saving model to output/0.7537.hdf5
    Epoch 5/20
    3085/3085 - 18s - loss: 0.5260 - val_loss: 0.7716
    
    Epoch 00005: val_loss did not improve from 0.75375
    Epoch 6/20
    3085/3085 - 18s - loss: 0.5241 - val_loss: 0.7664
    
    Epoch 00006: val_loss did not improve from 0.75375
    Epoch 7/20
    3085/3085 - 18s - loss: 0.5195 - val_loss: 0.7563
    
    Epoch 00007: val_loss did not improve from 0.75375
    Epoch 8/20
    3085/3085 - 17s - loss: 0.5117 - val_loss: 0.7111
    
    Epoch 00008: val_loss improved from 0.75375 to 0.71105, saving model to output/0.7111.hdf5
    Epoch 9/20
    3085/3085 - 18s - loss: 0.5047 - val_loss: 0.7616
    
    Epoch 00009: val_loss did not improve from 0.71105
    Epoch 10/20
    3085/3085 - 18s - loss: 0.4988 - val_loss: 0.6779
    
    Epoch 00010: val_loss improved from 0.71105 to 0.67790, saving model to output/0.6779.hdf5
    Epoch 11/20
    3085/3085 - 18s - loss: 0.4954 - val_loss: 0.6436
    
    Epoch 00011: val_loss improved from 0.67790 to 0.64363, saving model to output/0.6436.hdf5
    Epoch 12/20
    3085/3085 - 18s - loss: 0.4913 - val_loss: 0.6797
    
    Epoch 00012: val_loss did not improve from 0.64363
    Epoch 13/20
    3085/3085 - 18s - loss: 0.4846 - val_loss: 0.6416
    
    Epoch 00013: val_loss improved from 0.64363 to 0.64158, saving model to output/0.6416.hdf5
    Epoch 14/20
    3085/3085 - 18s - loss: 0.4808 - val_loss: 0.6461
    
    Epoch 00014: val_loss did not improve from 0.64158
    Epoch 15/20
    3085/3085 - 18s - loss: 0.4785 - val_loss: 0.6554
    
    Epoch 00015: val_loss did not improve from 0.64158
    Epoch 16/20
    3085/3085 - 18s - loss: 0.4765 - val_loss: 0.5874
    
    Epoch 00016: val_loss improved from 0.64158 to 0.58741, saving model to output/0.5874.hdf5
    Epoch 17/20
    3085/3085 - 18s - loss: 0.4742 - val_loss: 0.6052
    
    Epoch 00017: val_loss did not improve from 0.58741
    Epoch 18/20
    3085/3085 - 18s - loss: 0.4734 - val_loss: 0.5957
    
    Epoch 00018: val_loss did not improve from 0.58741
    Epoch 19/20
    3085/3085 - 17s - loss: 0.4713 - val_loss: 0.6707
    
    Epoch 00019: val_loss did not improve from 0.58741
    Epoch 20/20
    3085/3085 - 18s - loss: 0.4698 - val_loss: 0.6393
    
    Epoch 00020: val_loss did not improve from 0.58741





    <tensorflow.python.keras.callbacks.History at 0x7f4caca79790>




```python
nn.load_weights('output/0.5874.hdf5') # Import the least val_loss
```

# Prediction


```python
prepared_test = add_neighbours(test, k=3)
X_test = prepared_test.drop(['data_ind'], axis=1)
```

    100%|██████████| 11/11 [00:44<00:00,  4.05s/it]



```python
X_test_norm = transformer.transform(X_test.fillna(0))
```


```python
X_test_norm[:5]
```




    array([[-0.00300835, -0.00545834,  0.00972185,  0.00681656,  0.59126395,
             0.59333384,  0.3890069 ,  0.3820696 ,  0.        ,  0.02014557,
             0.00245   ,  0.00290529, -0.00206989,  0.00693733, -0.02014557],
           [-0.00252528, -0.00426304, -0.00411378,  0.00513677,  0.44063714,
             0.4414619 ,  0.5499566 ,  0.55488485,  0.        ,  0.01448405,
             0.00173776, -0.00925055, -0.00082474, -0.00492826, -0.01448405],
           [-0.00225754, -0.00130645,  0.00582741,  0.00499869,  0.40678105,
             0.4056811 ,  0.5793259 ,  0.57797784,  0.        ,  0.0101163 ,
            -0.00095108,  0.00082872,  0.00109993,  0.00134809, -0.0101163 ],
           [-0.00126105, -0.00467851, -0.00375272,  0.00103273,  0.2839596 ,
             0.28194952,  0.64870447,  0.6470427 ,  0.        ,  0.0124427 ,
             0.00341746, -0.00478544,  0.00201007,  0.00166179, -0.0124427 ],
           [-0.00162928,  0.00094605,  0.00400398,  0.00541683,  0.4728925 ,
             0.47000167,  0.5251422 ,  0.5286355 ,  0.        ,  0.00903365,
            -0.00257533, -0.00141285,  0.00289081, -0.00349329, -0.00903365]],
          dtype=float32)




```python
probas = nn.predict_proba(X_test_norm)
```

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/sequential.py:425: UserWarning: `model.predict_proba()` is deprecated and will be removed after 2021-01-01. Please use `model.predict()` instead.
      warnings.warn('`model.predict_proba()` is deprecated and '



```python
print(probas.min())
print(probas.max())
print(probas.mean())
```

    3.867151e-19
    0.9560038
    0.3008662



```python
probas.shape
```




    (20179459, 1)




```python
probas = np.squeeze(probas)
```


```python
df = pd.DataFrame({'id': prepared_test.index, 'signal': probas}).groupby('id')
agg = df.aggregate(('mean')).loc[:, ['signal']]
```


```python
agg.shape
```




    (7698899, 1)




```python
agg.head()
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
      <th>signal</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.280156</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001745</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.563104</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.310427</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.009493</td>
    </tr>
  </tbody>
</table>
</div>


