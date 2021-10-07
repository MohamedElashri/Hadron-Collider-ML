
# Measuring the Z boson mass


Let's look at a sample of $Z$ boson candidates recorded by CMS in 2011 and published at CERN opendata portal. It comes from DoubleMuon dataset with the following selection applied:

- Both muons are "global" muons
- invariant mass sits in range: 60 GeV < $ M_{\mu\mu}$ < 120 GeV
- |$\eta$| < 2.1 for both muons
- $p_{t}$ > 20 GeV

The following columns presented in the CSV file:

- `Run`, Event are the run and event numbers, respectively
- `pt` is the transverse momentum $p_{t}$ of the muon
- `eta` is the pseudorapidity of the muon: $\eta$
- `phi` is the $\phi$ angle of the muon direction
- `Q` is the charge of the muon
- `dxy` is the impact parameter in the transverse plane: $d_{xy}$ - how distant is the track from the collision point
- `iso` is the track isolation: $I_{track}$ - how many other tracks are there aroung given track


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
%matplotlib inline
```

## Read dataset


```python
df = pd.read_csv('./Zmumu.csv')
df.shape
```




    (10000, 14)



Let's calculate the invariant mass $M$ of the two muons using the formula

$M = \sqrt{2p_{t}^{1}p_{t}^{2}(\cosh(\eta_{1}-\eta_{2}) - \cos(\phi_{1}-\phi_{2}))}$


```python
df['M'] = np.sqrt(2 * df['pt1'] * df['pt2'] * (np.cosh(df['eta1'] - df['eta2']) - np.cos(df['phi1'] - df['phi2'])))
df.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Run</th>
      <th>Event</th>
      <th>pt1</th>
      <th>eta1</th>
      <th>phi1</th>
      <th>Q1</th>
      <th>dxy1</th>
      <th>iso1</th>
      <th>pt2</th>
      <th>eta2</th>
      <th>phi2</th>
      <th>Q2</th>
      <th>dxy2</th>
      <th>iso2</th>
      <th>M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165617</td>
      <td>74969122</td>
      <td>54.7055</td>
      <td>-0.4324</td>
      <td>2.5742</td>
      <td>1</td>
      <td>-0.0745</td>
      <td>0.4999</td>
      <td>34.2464</td>
      <td>-0.9885</td>
      <td>-0.4987</td>
      <td>-1</td>
      <td>0.0712</td>
      <td>3.4221</td>
      <td>89.885919</td>
    </tr>
    <tr>
      <th>1</th>
      <td>165617</td>
      <td>75138253</td>
      <td>24.5872</td>
      <td>-2.0522</td>
      <td>2.8666</td>
      <td>-1</td>
      <td>-0.0554</td>
      <td>0.0000</td>
      <td>28.5389</td>
      <td>0.3852</td>
      <td>-1.9912</td>
      <td>1</td>
      <td>0.0515</td>
      <td>0.0000</td>
      <td>88.812177</td>
    </tr>
  </tbody>
</table>
</div>



# Model

The distribution of the Z boson mass has the form of a normal distribution, in addition there is a noise. The distribution of the noise has an exponential form. Thus, the resulting model, it is a result of a superposition of two distributions - normal and polinomial

Let's plot the distribution of Z boson mass


```python
def plot_mass(mass, bins_count=100):
    y, x = np.histogram(mass, bins=bins_count, density=False)
    err = np.sqrt(y)

    fig = plt.figure(figsize=(15,7))
    plt.title('Z mass', fontsize=20)
    plt.xlabel("$m_{\mu\mu}$ [GeV]", fontsize=20)
    plt.ylabel("Number of events", fontsize=20)
    plt.errorbar(x[:-1], y, yerr=err, fmt='o', color='red', ecolor='grey', capthick=0.5, zorder=1, label="data")
    return y, x
```


```python
plot_mass(df.M);
```


![png](index-checkpoint_files/index-checkpoint_9_0.png)


## Exercise 1. clean up dataset a bit
- demand that charge of muons should be opposite
- $I_{track}$ < 3 and $d_{xy}$ < 0.2 cm


```python
### YOUR CODE GOES HERE ###
df_sign = None #
df_isolation = None #
df = df
```


```python
plot_mass(df.M);
```


![png](index-checkpoint_files/index-checkpoint_12_0.png)


### Let's define parametrised model
it should represent mixture of 1) Gaussian signal and 2) background that for the simplicity we consider to be flat over mass. So it gives the following set of parameters:

- m0 - center of the Gaussian
- sigma - standard deviation of the Gaussian
- ampl - height of the peak
- bck - height of the background 

finding those parameters is called _fitting_ model into the data. It will be the goal for the rest of the exercise. For simplicity sake we'll stick with old good binned fit.


```python
def model_predict(params, X):
    m0, sigma, ampl, bck = params
    return bck + ampl / (sigma * np.sqrt(2 * np.pi)) * np.exp((-1) * (X - m0)**2 / (2 * sigma**2))
```


```python
def model_loss(params, X, y):
#     y, x = np.histogram(mass, bins=bins_count, density=False)
#     residuals = model_predict(params, (x[1:] + x[:-1])/2) - y 
    residuals = y - model_predict(params, X)
    return np.sum(residuals**2) / len(residuals)
```


```python
def plot_mass_with_model(params, mass, bins_count=100):
    y, X = plot_mass(mass, bins_count=bins_count)
    X = (X[1:] + X[:-1]) / 2
    error = model_loss(params, X, y)
    plt.plot(X, model_predict(params, X), color='blue', linewidth=3.0, zorder=2, label="fit, loss=%.2f" % error)
    plt.legend(fontsize='x-large')
```

## Here you can fit model parameters by hand


```python
plot_mass_with_model((91, 2.2, 4600, 15), df.M)
```


![png](index-checkpoint_files/index-checkpoint_18_0.png)


## ... but you can do it automatically of course

Setting up a scikit optimizer


```python
from tqdm import tqdm
from skopt import Optimizer

search_space = [(80, 100), # m0 range
    (1.0, 3.0), # sigma range
    (350, 450), # amplitude range
    (0, 10) # bck range
    ]
y, X = np.histogram(df.M, bins=120, density=False)
X = (X[1:] + X[:-1]) / 2
opt = Optimizer(search_space, base_estimator="GP", acq_func="EI", acq_optimizer="lbfgs")
```

Running it for a while. You can re-run this cell several times


```python
from skopt.utils import create_result
for i in tqdm(range(50)):
    next_x = opt.ask()
    f_val = model_loss(next_x, X, y)
    opt.tell(next_x, f_val)
    
res = create_result(Xi=opt.Xi, yi=opt.yi, space=opt.space,
                         rng=opt.rng, models=opt.models)
```

    100%|██████████| 50/50 [00:22<00:00,  2.26it/s]


## A bit of search history


```python
import skopt.plots
skopt.plots.plot_convergence(res)
print (list(zip(["m0", "sigma", "ampl", "bck"], res.x)))
```

    [('m0', 91), ('sigma', 2), ('ampl', 4165), ('bck', 3)]



![png](index-checkpoint_files/index-checkpoint_25_1.png)



```python
# even more details on the search space
# skopt.plots.plot_objective(res, dimensions=['m0', 'sigma', 'ampl', 'bck'])

```

Let's see how well the prediction fits the data


```python
plot_mass_with_model(res.x, df.M, bins_count=120)
```


![png](index-checkpoint_files/index-checkpoint_28_0.png)



```python
# token expires every 30 min
COURSERA_TOKEN = "aAja9pbfyRMtibne"### YOUR TOKEN HERE
COURSERA_EMAIL = "andrey.u@gmail.com"### YOUR EMAIL HERE
```

## Grader part, do not change, please


```python
import grading
grader = grading.Grader(assignment_key="SeMsm2JfEeiyvBKA0wfIgA", 
                      all_parts=["VI3xu", "VuE8x", "KzmMV", "TwZBF"])

```


```python
ans_part1 = round(res.x[0])
grader.set_answer("VI3xu", ans_part1)

ans_part2 = round(res.x[1], 2)
grader.set_answer("VuE8x", ans_part2)

ans_part3 = round(res.x[3])
grader.set_answer("KzmMV", ans_part3)

ans_part4 = round(res.x[2])
grader.set_answer("TwZBF", ans_part4)

grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)
```

    Submitted to Coursera platform. See results on assignment page!

