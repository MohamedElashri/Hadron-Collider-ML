
# About

In this programming assignment you will optimize a simple tracking system using Bayesian optimization with Gaussian processes. The same method can be used for more complex systems optimization. For an example, the ATLAS tracking system:

<img src="pic/tracks.png" width="700" />
https://twiki.cern.ch/twiki/bin/view/AtlasPublic/EventDisplayRun2Collisions


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
```

# Geometry generation

Our simple tracking system consists of 4 layers of straw tubes, which detects particles fly through them. Each layer has 200 tubes. The system is parametrized by six adjustable parameters: $y_{1}$, $y_{2}$, $y_{3}$, $z_{1}$, $z_{2}$, $z_{3}$. These parameters describe shifts between the layers as it is show in the figure: 

<img src="pic/system.png" alt="Drawing" style="width: 700px;">

In this notebook we suppose that the radius $R$ of tubes is 1 cm and the distance between tubes in one layer (pitch) is 4 cm. We take these values as constants and will not change them.  Also z-value must be more than 2, otherwise these tubes will intersect. It's wrong.

For an example, lets select values for the layers shifts.


```python
# CONSTANT PARAMETERS
R = 1. # cm
pitch = 4.0 #cm 

# ADJUSTABLE PARAMETERS
y1 = 0.0
y2 = 0.0
y3 = 0.0
z1 = 2.0
z2 = 4.0
z3 = 6.0
```

Using these parameter values we generate $Z$ and $Y$ coordinates of tubes in the tracking system.


```python
tr = utils.Tracker(R, pitch, y1, y2, y3, z1, z2, z3)
Z, Y = tr.create_geometry()
```

Now display several tubes of the generated geometry.


```python
utils.geometry_display(Z, Y, R, y_min=-10, y_max=10)
```


![png](tracker-checkpoint_files/tracker-checkpoint_8_0.png)


# Tracks generation

Lets generate several tracks of particles fly in the tracking system. We consider straight tracks with equation: 

$$
y = kz + b
$$

where $z, y$ are coordinates of the track, $k$ is slope of the track and $b$ is the y intercept.

Track parameters are generated from the following distributions:

$$
b \in U(b_{min}, b_{max})\\
k = tan(\alpha), \alpha \in N(\mu_{\alpha}, \sigma_{\alpha})
$$

where $U$ is uniform distribution and $N$ is normal distribution.


```python
N_tracks = 1000
t = utils.Tracks(b_min=-100, b_max=100, alpha_mean=0, alpha_std=0.2)
tracks = t.generate(N_tracks)
```

Display the tubes geometry with the generated tracks.


```python
utils.geometry_display(Z, Y, R, y_min=-10, y_max=10)
utils.tracks_display(tracks, Z)
```


![png](tracker-checkpoint_files/tracker-checkpoint_12_0.png)


# Target metric

For a given geaometry of the tracking system we will calculate the ratio of tracks with at least 2 hits to the total number of tracks:

$$Score = \frac{N\_tracks_{n\_hits \ge 2}}{N\_tracks}$$

The higher score, the better.


```python
score = utils.get_score(Z, Y, tracks, R)
print(score)
```

    0.648


# Optimization

In this programming assignment you need to find parameters of geometry of the tracking system that provides the highest **score** value. However, we propose to solve a minimization problem. So you need to minimize **1-score** value. This is an objective function of the optimization


Lets define ranges of the adjustable parameters of the system.


```python
y1_min, y1_max = [0, 4]
y2_min, y2_max = [0, 4]
y3_min, y3_max = [0, 4]
z1_min, z1_max = [2, 10]
z2_min, z2_max = [2, 10]
z3_min, z3_max = [2, 10]
```

And generate tracks used during the optimization.


```python
t = utils.Tracks(-100, 100, 0, 0.2)
tracks = t.generate(1000)
```

Define the objective function of the optimization


```python
def objective(x):
    
    R, pitch, y1, y2, y3, z1, z2, z3 = x
    Z, Y = utils.Tracker(R, pitch, y1, y2, y3, z1, z2, z3).create_geometry()
    val = utils.get_score(Z, Y, tracks, R)
    
    return 1. - val # the smaller, the better.
```

## Grid search

Firstly, lets try to solve the optimization problem using grid search.

TASK: find optimal parameters of the tracking system using grid search.


```python
%%time

# Number of unique values for each of the adjustable parameters.
n_points = 2

# Define grid of the parameters
y1_grid = np.linspace(y1_min, y2_max, n_points)
y2_grid = np.linspace(y2_min, y2_max, n_points)
y3_grid = np.linspace(y3_min, y3_max, n_points)
z1_grid = np.linspace(z1_min, z1_max, n_points)
z2_grid = np.linspace(z2_min, z2_max, n_points)
z3_grid = np.linspace(z3_min, z3_max, n_points)

# Define list to store the optimization results
min_objective_values = []
params_for_min_objective_values = []

is_first = True

# Loop on the grid
for y1 in y1_grid:
    for y2 in y2_grid:
        for y3 in y3_grid:
            for z1 in z1_grid:
                for z2 in z2_grid:
                    for z3 in z3_grid:

                        # Calculate the objective function value for a grid node
                        x = [R, pitch, y1, y2, y3, z1, z2, z3]
                        val = objective(x)

                        if is_first:
                            min_objective_values.append(val)
                            params_for_min_objective_values.append(tuple(x))
                            is_first = False
                        elif val < min_objective_values[-1]:
                            min_objective_values.append(val)
                            params_for_min_objective_values.append(tuple(x))
                        else:
                            min_objective_values.append(min_objective_values[-1])
                            params_for_min_objective_values.append(params_for_min_objective_values[-1])
```

    CPU times: user 1.1 s, sys: 4.98 ms, total: 1.11 s
    Wall time: 1.13 s


Plot the optimization curve


```python
print("Objective optimum = ", min_objective_values[-1])
utils.plot_objective(min_objective_values)
```

    ('Objective optimum = ', 0.399)



![png](tracker-checkpoint_files/tracker-checkpoint_24_1.png)



```python
(R, pitch, y1, y2, y3, z1, z2, z3) = params_for_min_objective_values[-1]
print("Optimal parameters: ")
print("y1 = ", y1)
print("y2 = ", y2)
print("y3 = ", y3)
print("z1 = ", z1)
print("z2 = ", z2)
print("z3 = ", z3)
```

    Optimal parameters: 
    ('y1 = ', 0.0)
    ('y2 = ', 0.0)
    ('y3 = ', 0.0)
    ('z1 = ', 2.0)
    ('z2 = ', 2.0)
    ('z3 = ', 2.0)


Display the optimal tracking system geometry.


```python
Z, Y = utils.Tracker(R, pitch, y1, y2, y3, z1, z2, z3).create_geometry()
utils.geometry_display(Z, Y, R, y_min=-10, y_max=10)
```


![png](tracker-checkpoint_files/tracker-checkpoint_27_0.png)


## Random search

Now, lets modify grid search. For this we will generate random points in the parameter space instead of the grid.

TASK: find optimal parameters of the tracking system using random search.


```python
%%time

# Number of random point to generate.
n_points = 100

# Define random values on the tracking system parameters
y1_grid = np.random.RandomState(12).uniform(y1_min, y1_max, n_points)
y2_grid = np.random.RandomState(13).uniform(y2_min, y2_max, n_points)
y3_grid = np.random.RandomState(14).uniform(y3_min, y3_max, n_points)
z1_grid = np.random.RandomState(15).uniform(z1_min, z1_max, n_points)
z2_grid = np.random.RandomState(16).uniform(z2_min, z2_max, n_points)
z3_grid = np.random.RandomState(17).uniform(z3_min, z3_max, n_points)

# Define list to store the optimization results
min_objective_values = []
params_for_min_objective_values = []

for i in range(n_points):
    
    y1 = y1_grid[i]
    y2 = y2_grid[i]
    y3 = y3_grid[i]
    z1 = z1_grid[i]
    z2 = z2_grid[i]
    z3 = z3_grid[i]
    
    # Calculate the objective function value for a grid node
    x = [R, pitch, y1, y2, y3, z1, z2, z3]
    val = objective(x)
    
    if i==0:
        min_objective_values.append(val)
        params_for_min_objective_values.append(tuple(x))
    elif val < min_objective_values[-1]:
        min_objective_values.append(val)
        params_for_min_objective_values.append(tuple(x))
    else:
        min_objective_values.append(min_objective_values[-1])
        params_for_min_objective_values.append(params_for_min_objective_values[-1])
```

    CPU times: user 1.73 s, sys: 9.25 ms, total: 1.74 s
    Wall time: 1.76 s


Plot the optimization curve


```python
print("Objective optimum = ", min_objective_values[-1])
utils.plot_objective(min_objective_values)
```

    ('Objective optimum = ', 0.399)



![png](tracker-checkpoint_files/tracker-checkpoint_31_1.png)



```python
(R, pitch, y1, y2, y3, z1, z2, z3) = params_for_min_objective_values[-1]
print("Optimal parameters: ")
print("y1 = ", y1)
print("y2 = ", y2)
print("y3 = ", y3)
print("z1 = ", z1)
print("z2 = ", z2)
print("z3 = ", z3)
```

    Optimal parameters: 
    ('y1 = ', 0.0)
    ('y2 = ', 0.0)
    ('y3 = ', 0.0)
    ('z1 = ', 2.0)
    ('z2 = ', 2.0)
    ('z3 = ', 2.0)


Display the optimal tracking system geometry.


```python
Z, Y = utils.Tracker(R, pitch, y1, y2, y3, z1, z2, z3).create_geometry()
utils.geometry_display(Z, Y, R, y_min=-10, y_max=10)
```


![png](tracker-checkpoint_files/tracker-checkpoint_34_0.png)


## Bayesian optimization with Gaussian processes

At this step we will use Bayesian optimization implemented using [scikit-optimize](http://scikit-optimize.github.io/) library to find optimal tracking system geometry. During the optimization Lower Confidence Bound (LCB) acquisition function is used:

$$
LCB(x) = \mu(x) - \kappa(x) 
$$

where $\kappa$ is adjustable parameter that defines the exploration-exploitation trade-off of the optimization.

TASK: find optimal parameters of the tracking system.


```python
from skopt import gp_minimize

y1_min, y1_max = [0, 1]
y2_min, y2_max = [0, 1]
y3_min, y3_max = [0, 1]
z1_min, z1_max = [2, 10]
z2_min, z2_max = [2, 10]
z3_min, z3_max = [2, 10]

kappa= 10.1
dimentions = [(R, R + 10**-6), (pitch, pitch + 10**-6), 
              (y1_min, y1_max), (y2_min, y2_max), (y3_min, y3_max),
              (z1_min, z1_max), (z2_min, z2_max), (z3_min, z3_max)]

res = gp_minimize(func=objective,                   # the function to minimize
                  dimensions=dimentions,            # the bounds on each dimension of x
                  acq_func="LCB",                   # the acquisition function
                  n_calls=100,                      # the number of evaluations of f 
                  n_random_starts=50,               # the number of random initialization points
                  noise=0.01**2,                    # the noise level (optional)
                  random_state=123,                 # the random seed
                  kappa=kappa,                      # the adjustable parameter of LCB
                  n_jobs=3)
```

Plot the optimization curve


```python
from skopt.plots import plot_convergence
print("Objective optimum = ", res.fun)
plot_convergence(res);
# 0.09399999999999997
```

    ('Objective optimum = ', 0.20499999999999996)



![png](tracker-checkpoint_files/tracker-checkpoint_38_1.png)


Display the optimal tracking system geometry.


```python
Z, Y = utils.Tracker(R, pitch, y1, y2, y3, z1, z2, z3).create_geometry()
utils.geometry_display(Z, Y, R, y_min=-10, y_max=10)
```


![png](tracker-checkpoint_files/tracker-checkpoint_40_0.png)


# Submission

TASK: Using the optimization methods above try to find the best geometry of the tracking system. Then, create sumbmission file with the parameters of your geometry and submit it.


```python
best = [R, pitch, y1, y2, y3, z1, z2, z3]
utils.create_solution(best, filename='submission_file.csv')
```




<a href='submission_file.csv' target='_blank'>submission_file.csv</a><br>




```python

```
