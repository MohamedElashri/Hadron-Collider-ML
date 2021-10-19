
<a href="https://colab.research.google.com/github/MohamedElashri/Hadron-Collider-ML/blob/master/Particle_identification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# About

In this programming assignment you will train a classifier to identify type of a particle. There are six particle types: electron, proton, muon, kaon, pion and ghost. Ghost is a particle with other type than the first five or a detector noise. 

Different particle types remain different responses in the detector systems or subdetectors. Thre are five systems: tracking system, ring imaging Cherenkov detector (RICH), electromagnetic and hadron calorimeters, and muon system.

![pid](https://github.com/MohamedElashri/Hadron-Collider-ML/blob/master/week2/pic/pid.jpg?raw=1)

You task is to identify a particle type using the responses in the detector systems.


```python
!wget https://raw.githubusercontent.com/MohamedElashri/Hadron-Collider-ML/master/week2/utils.py
```

    --2021-03-01 17:26:35--  https://raw.githubusercontent.com/MohamedElashri/Hadron-Collider-ML/master/week2/utils.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7634 (7.5K) [text/plain]
    Saving to: ‘utils.py.1’
    
    utils.py.1          100%[===================>]   7.46K  --.-KB/s    in 0s      
    
    2021-03-01 17:26:35 (80.6 MB/s) - ‘utils.py.1’ saved [7634/7634]
    



```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import utils
```

# Download data

Download data used to train classifiers.

### Read training file


```python
!wget https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/data/test.csv.gz
!wget https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/data/training.csv.gz
```

    --2021-03-01 17:26:35--  https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/data/test.csv.gz
    Resolving github.com (github.com)... 140.82.112.3
    Connecting to github.com (github.com)|140.82.112.3|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://github-releases.githubusercontent.com/187530009/ea4d2f80-7a88-11eb-866a-4e2953dcf3d7?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T172635Z&X-Amz-Expires=300&X-Amz-Signature=c509f869d64564e547790cda7707cc2bd4d4c58bd3717fc352d12e1411342cb0&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dtest.csv.gz&response-content-type=application%2Foctet-stream [following]
    --2021-03-01 17:26:35--  https://github-releases.githubusercontent.com/187530009/ea4d2f80-7a88-11eb-866a-4e2953dcf3d7?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T172635Z&X-Amz-Expires=300&X-Amz-Signature=c509f869d64564e547790cda7707cc2bd4d4c58bd3717fc352d12e1411342cb0&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dtest.csv.gz&response-content-type=application%2Foctet-stream
    Resolving github-releases.githubusercontent.com (github-releases.githubusercontent.com)... 185.199.111.154, 185.199.109.154, 185.199.108.154, ...
    Connecting to github-releases.githubusercontent.com (github-releases.githubusercontent.com)|185.199.111.154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 248364236 (237M) [application/octet-stream]
    Saving to: ‘test.csv.gz.1’
    
    test.csv.gz.1       100%[===================>] 236.86M  36.7MB/s    in 7.2s    
    
    2021-03-01 17:26:42 (32.7 MB/s) - ‘test.csv.gz.1’ saved [248364236/248364236]
    
    --2021-03-01 17:26:42--  https://github.com/MohamedElashri/Hadron-Collider-ML/releases/download/data/training.csv.gz
    Resolving github.com (github.com)... 140.82.112.4
    Connecting to github.com (github.com)|140.82.112.4|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://github-releases.githubusercontent.com/187530009/d0abe800-7a88-11eb-9b83-b9573f25cb2a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T172643Z&X-Amz-Expires=300&X-Amz-Signature=22b2017f3a8821e3fe521431d7bc3116276f1542d0a7a0f215f66b8c48a45271&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dtraining.csv.gz&response-content-type=application%2Foctet-stream [following]
    --2021-03-01 17:26:43--  https://github-releases.githubusercontent.com/187530009/d0abe800-7a88-11eb-9b83-b9573f25cb2a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210301T172643Z&X-Amz-Expires=300&X-Amz-Signature=22b2017f3a8821e3fe521431d7bc3116276f1542d0a7a0f215f66b8c48a45271&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=187530009&response-content-disposition=attachment%3B%20filename%3Dtraining.csv.gz&response-content-type=application%2Foctet-stream
    Resolving github-releases.githubusercontent.com (github-releases.githubusercontent.com)... 185.199.108.154, 185.199.109.154, 185.199.110.154, ...
    Connecting to github-releases.githubusercontent.com (github-releases.githubusercontent.com)|185.199.108.154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 247314656 (236M) [application/octet-stream]
    Saving to: ‘training.csv.gz.1’
    
    training.csv.gz.1   100%[===================>] 235.86M  41.9MB/s    in 5.4s    
    
    2021-03-01 17:26:48 (44.0 MB/s) - ‘training.csv.gz.1’ saved [247314656/247314656]
    



```python
data = pandas.read_csv('training.csv.gz')
```


```python
data.head()
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
      <th>TrackP</th>
      <th>TrackNDoFSubdetector2</th>
      <th>BremDLLbeElectron</th>
      <th>MuonLooseFlag</th>
      <th>FlagSpd</th>
      <th>SpdE</th>
      <th>EcalDLLbeElectron</th>
      <th>DLLmuon</th>
      <th>RICHpFlagElectron</th>
      <th>EcalDLLbeMuon</th>
      <th>TrackQualitySubdetector2</th>
      <th>FlagPrs</th>
      <th>DLLelectron</th>
      <th>DLLkaon</th>
      <th>EcalE</th>
      <th>TrackQualityPerNDoF</th>
      <th>DLLproton</th>
      <th>PrsDLLbeElectron</th>
      <th>FlagRICH1</th>
      <th>MuonLLbeBCK</th>
      <th>FlagHcal</th>
      <th>EcalShowerLongitudinalParameter</th>
      <th>Calo2dFitQuality</th>
      <th>TrackPt</th>
      <th>TrackDistanceToZ</th>
      <th>RICHpFlagPion</th>
      <th>HcalDLLbeElectron</th>
      <th>Calo3dFitQuality</th>
      <th>FlagEcal</th>
      <th>MuonLLbeMuon</th>
      <th>TrackNDoFSubdetector1</th>
      <th>RICHpFlagProton</th>
      <th>RICHpFlagKaon</th>
      <th>GhostProbability</th>
      <th>TrackQualitySubdetector1</th>
      <th>Label</th>
      <th>RICH_DLLbeBCK</th>
      <th>FlagRICH2</th>
      <th>FlagBrem</th>
      <th>HcalDLLbeMuon</th>
      <th>TrackNDoF</th>
      <th>RICHpFlagMuon</th>
      <th>RICH_DLLbeKaon</th>
      <th>RICH_DLLbeElectron</th>
      <th>HcalE</th>
      <th>MuonFlag</th>
      <th>FlagMuon</th>
      <th>PrsE</th>
      <th>RICH_DLLbeMuon</th>
      <th>RICH_DLLbeProton</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>74791.156263</td>
      <td>15.0</td>
      <td>0.232275</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.2</td>
      <td>-2.505719</td>
      <td>6.604153</td>
      <td>1.0</td>
      <td>1.929960</td>
      <td>17.585680</td>
      <td>1.0</td>
      <td>-6.411697</td>
      <td>-7.213295</td>
      <td>0.000001</td>
      <td>1.467550</td>
      <td>-26.667494</td>
      <td>-2.730674</td>
      <td>1.0</td>
      <td>-5.152923</td>
      <td>1.0</td>
      <td>-999.000000</td>
      <td>19.954819</td>
      <td>3141.930677</td>
      <td>0.613640</td>
      <td>1.0</td>
      <td>-0.909544</td>
      <td>-999.000000</td>
      <td>1.0</td>
      <td>-0.661823</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.018913</td>
      <td>5.366212</td>
      <td>Muon</td>
      <td>-21.913000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.015345</td>
      <td>28.0</td>
      <td>1.0</td>
      <td>-7.213300</td>
      <td>-0.280200</td>
      <td>5586.589846</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>10.422315</td>
      <td>-2.081143e-07</td>
      <td>-24.824400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2738.489989</td>
      <td>15.0</td>
      <td>-0.357748</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.2</td>
      <td>1.864351</td>
      <td>0.263651</td>
      <td>1.0</td>
      <td>-2.061959</td>
      <td>20.230680</td>
      <td>1.0</td>
      <td>5.453014</td>
      <td>0.000006</td>
      <td>1531.542000</td>
      <td>3.570540</td>
      <td>-0.711194</td>
      <td>1.773806</td>
      <td>1.0</td>
      <td>-999.000000</td>
      <td>0.0</td>
      <td>33.187644</td>
      <td>0.037601</td>
      <td>199.573653</td>
      <td>0.465480</td>
      <td>1.0</td>
      <td>0.434909</td>
      <td>13.667366</td>
      <td>1.0</td>
      <td>-999.000000</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.351206</td>
      <td>9.144749</td>
      <td>Ghost</td>
      <td>-0.703617</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-2.394644</td>
      <td>32.0</td>
      <td>1.0</td>
      <td>-0.324317</td>
      <td>1.707283</td>
      <td>-0.000007</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>43.334935</td>
      <td>2.771583e+00</td>
      <td>-0.648017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2161.409908</td>
      <td>17.0</td>
      <td>-999.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-999.0</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>11.619878</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>0.826442</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>94.829418</td>
      <td>0.241891</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.195717</td>
      <td>1.459992</td>
      <td>Ghost</td>
      <td>-999.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>-999.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-999.000000</td>
      <td>-9.990000e+02</td>
      <td>-999.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15277.730490</td>
      <td>20.0</td>
      <td>-0.638984</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.2</td>
      <td>-2.533918</td>
      <td>-8.724949</td>
      <td>1.0</td>
      <td>-3.253981</td>
      <td>15.336305</td>
      <td>1.0</td>
      <td>-10.616585</td>
      <td>-39.447507</td>
      <td>4385.688000</td>
      <td>1.076721</td>
      <td>-29.291509</td>
      <td>-3.053104</td>
      <td>1.0</td>
      <td>-999.000000</td>
      <td>1.0</td>
      <td>231.190351</td>
      <td>2.839508</td>
      <td>808.631064</td>
      <td>0.680705</td>
      <td>1.0</td>
      <td>-1.504160</td>
      <td>1939.259641</td>
      <td>1.0</td>
      <td>-999.000000</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.003972</td>
      <td>22.950573</td>
      <td>Pion</td>
      <td>-47.223118</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-0.321242</td>
      <td>36.0</td>
      <td>1.0</td>
      <td>-35.202221</td>
      <td>-14.742319</td>
      <td>4482.803707</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.194175</td>
      <td>-3.070819e+00</td>
      <td>-29.291519</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7563.700195</td>
      <td>19.0</td>
      <td>-0.638962</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.2</td>
      <td>-2.087146</td>
      <td>-7.060422</td>
      <td>1.0</td>
      <td>-0.995816</td>
      <td>10.954629</td>
      <td>1.0</td>
      <td>-8.144945</td>
      <td>26.050386</td>
      <td>1220.930044</td>
      <td>0.439767</td>
      <td>21.386587</td>
      <td>-2.730648</td>
      <td>1.0</td>
      <td>-999.000000</td>
      <td>1.0</td>
      <td>-794.866475</td>
      <td>1.209193</td>
      <td>1422.569214</td>
      <td>0.575066</td>
      <td>1.0</td>
      <td>-1.576249</td>
      <td>1867.165142</td>
      <td>1.0</td>
      <td>-999.000000</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015232</td>
      <td>3.516173</td>
      <td>Proton</td>
      <td>15.304688</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.038026</td>
      <td>33.0</td>
      <td>1.0</td>
      <td>25.084287</td>
      <td>-10.272412</td>
      <td>5107.554680</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000015</td>
      <td>-5.373712e+00</td>
      <td>23.653087</td>
    </tr>
  </tbody>
</table>
</div>



### List of columns in the samples

Here, **Spd** stands for Scintillating Pad Detector, **Prs** - Preshower, **Ecal** - electromagnetic calorimeter, **Hcal** - hadronic calorimeter, **Brem** denotes traces of the particles that were deflected by detector.

- ID - id value for tracks (presents only in the test file for the submitting purposes)
- Label - string valued observable denoting particle types. Can take values "Electron", "Muon", "Kaon", "Proton", "Pion" and "Ghost". This column is absent in the test file.
- FlagSpd - flag (0 or 1), if reconstructed track passes through Spd
- FlagPrs - flag (0 or 1), if reconstructed track passes through Prs
- FlagBrem - flag (0 or 1), if reconstructed track passes through Brem
- FlagEcal - flag (0 or 1), if reconstructed track passes through Ecal
- FlagHcal - flag (0 or 1), if reconstructed track passes through Hcal
- FlagRICH1 - flag (0 or 1), if reconstructed track passes through the first RICH detector
- FlagRICH2 - flag (0 or 1), if reconstructed track passes through the second RICH detector
- FlagMuon - flag (0 or 1), if reconstructed track passes through muon stations (Muon)
- SpdE - energy deposit associated to the track in the Spd
- PrsE - energy deposit associated to the track in the Prs
- EcalE - energy deposit associated to the track in the Hcal
- HcalE - energy deposit associated to the track in the Hcal
- PrsDLLbeElectron - delta log-likelihood for a particle candidate to be electron using information from Prs
- BremDLLbeElectron - delta log-likelihood for a particle candidate to be electron using information from Brem
- TrackP - particle momentum
- TrackPt - particle transverse momentum
- TrackNDoFSubdetector1  - number of degrees of freedom for track fit using hits in the tracking sub-detector1
- TrackQualitySubdetector1 - chi2 quality of the track fit using hits in the tracking sub-detector1
- TrackNDoFSubdetector2 - number of degrees of freedom for track fit using hits in the tracking sub-detector2
- TrackQualitySubdetector2 - chi2 quality of the track fit using hits in the  tracking sub-detector2
- TrackNDoF - number of degrees of freedom for track fit using hits in all tracking sub-detectors
- TrackQualityPerNDoF - chi2 quality of the track fit per degree of freedom
- TrackDistanceToZ - distance between track and z-axis (beam axis)
- Calo2dFitQuality - quality of the 2d fit of the clusters in the calorimeter 
- Calo3dFitQuality - quality of the 3d fit in the calorimeter with assumption that particle was electron
- EcalDLLbeElectron - delta log-likelihood for a particle candidate to be electron using information from Ecal
- EcalDLLbeMuon - delta log-likelihood for a particle candidate to be muon using information from Ecal
- EcalShowerLongitudinalParameter - longitudinal parameter of Ecal shower
- HcalDLLbeElectron - delta log-likelihood for a particle candidate to be electron using information from Hcal
- HcalDLLbeMuon - delta log-likelihood for a particle candidate to be using information from Hcal
- RICHpFlagElectron - flag (0 or 1) if momentum is greater than threshold for electrons to produce Cherenkov light
- RICHpFlagProton - flag (0 or 1) if momentum is greater than threshold for protons to produce Cherenkov light
- RICHpFlagPion - flag (0 or 1) if momentum is greater than threshold for pions to produce Cherenkov light
- RICHpFlagKaon - flag (0 or 1) if momentum is greater than threshold for kaons to produce Cherenkov light
- RICHpFlagMuon - flag (0 or 1) if momentum is greater than threshold for muons to produce Cherenkov light
- RICH_DLLbeBCK  - delta log-likelihood for a particle candidate to be background using information from RICH
- RICH_DLLbeKaon - delta log-likelihood for a particle candidate to be kaon using information from RICH
- RICH_DLLbeElectron - delta log-likelihood for a particle candidate to be electron using information from RICH
- RICH_DLLbeMuon - delta log-likelihood for a particle candidate to be muon using information from RICH
- RICH_DLLbeProton - delta log-likelihood for a particle candidate to be proton using information from RICH
- MuonFlag - muon flag (is this track muon) which is determined from muon stations
- MuonLooseFlag muon flag (is this track muon) which is determined from muon stations using looser criteria
- MuonLLbeBCK - log-likelihood for a particle candidate to be not muon using information from muon stations
- MuonLLbeMuon - log-likelihood for a particle candidate to be muon using information from muon stations
- DLLelectron - delta log-likelihood for a particle candidate to be electron using information from all subdetectors
- DLLmuon - delta log-likelihood for a particle candidate to be muon using information from all subdetectors
- DLLkaon - delta log-likelihood for a particle candidate to be kaon using information from all subdetectors
- DLLproton - delta log-likelihood for a particle candidate to be proton using information from all subdetectors
- GhostProbability - probability for a particle candidate to be ghost track. This variable is an output of classification model used in the tracking algorithm.

Delta log-likelihood in the features descriptions means the difference between log-likelihood for the mass hypothesis that a given track is left by some particle (for example, electron) and log-likelihood for the mass hypothesis that a given track is left by a pion (so, DLLpion = 0 and thus we don't have these columns). This is done since most tracks (~80%) are left by pions and in practice we actually need to discriminate other particles from pions. In other words, the null hypothesis is that particle is a pion.

### Look at the labels set

The training data contains six classes. Each class corresponds to a particle type. Your task is to predict type of a particle.


```python
set(data.Label)
```




    {'Electron', 'Ghost', 'Kaon', 'Muon', 'Pion', 'Proton'}



Convert the particle types into class numbers.


```python
data['Class'] = utils.get_class_ids(data.Label.values)
set(data.Class)
```




    {0, 1, 2, 3, 4, 5}



### Define training features

The following set of features describe particle responses in the detector systems:

![features](https://github.com/MohamedElashri/Hadron-Collider-ML/blob/master/week2/pic/features.jpeg?raw=1)

Also there are several combined features. The full list is following.


```python
features = list(set(data.columns) - {'Label', 'Class'})
features
```




    ['RICH_DLLbeKaon',
     'PrsDLLbeElectron',
     'RICHpFlagElectron',
     'DLLproton',
     'RICHpFlagKaon',
     'FlagPrs',
     'TrackP',
     'HcalDLLbeMuon',
     'TrackQualitySubdetector2',
     'Calo2dFitQuality',
     'FlagMuon',
     'FlagHcal',
     'EcalDLLbeMuon',
     'RICHpFlagMuon',
     'FlagBrem',
     'EcalE',
     'HcalE',
     'BremDLLbeElectron',
     'DLLelectron',
     'MuonLooseFlag',
     'Calo3dFitQuality',
     'RICH_DLLbeBCK',
     'RICHpFlagProton',
     'MuonFlag',
     'GhostProbability',
     'RICH_DLLbeProton',
     'FlagSpd',
     'SpdE',
     'FlagEcal',
     'DLLmuon',
     'FlagRICH1',
     'MuonLLbeMuon',
     'PrsE',
     'EcalShowerLongitudinalParameter',
     'TrackPt',
     'TrackNDoF',
     'TrackNDoFSubdetector2',
     'DLLkaon',
     'TrackQualitySubdetector1',
     'RICH_DLLbeElectron',
     'TrackQualityPerNDoF',
     'RICHpFlagPion',
     'RICH_DLLbeMuon',
     'FlagRICH2',
     'TrackNDoFSubdetector1',
     'EcalDLLbeElectron',
     'HcalDLLbeElectron',
     'TrackDistanceToZ',
     'MuonLLbeBCK']



### Divide training data into 2 parts


```python
training_data, validation_data = train_test_split(data, random_state=11, train_size=0.10)
```


```python
len(training_data), len(validation_data)
```




    (120000, 1080000)



# Sklearn classifier

On this step your task is to train **Sklearn** classifier to provide lower **log loss** value.


TASK: your task is to tune the classifier parameters to achieve the lowest **log loss** value on the validation sample you can.


```python
from sklearn.ensemble import GradientBoostingClassifier
```


```python
%%time 
gb = GradientBoostingClassifier(learning_rate=0.05, n_estimators=1000, subsample=0.3, random_state=13,
                                min_samples_leaf=10, max_depth=30)
gb.fit(training_data[features].values, training_data.Class.values)
```

    CPU times: user 2h 18min 15s, sys: 5.02 s, total: 2h 18min 20s
    Wall time: 2h 18min 20s


### Log loss on the cross validation sample


```python
# predict each track
proba_gb = gb.predict_proba(validation_data[features].values)
```


```python
log_loss(validation_data.Class.values, proba_gb)
```




    1.0234072680151165



# Keras neural network

On this step your task is to train **Keras** NN classifier to provide lower **log loss** value.


TASK: your task is to tune the classifier parameters to achieve the lowest **log loss** value on the validation sample you can. Data preprocessing may help you to improve your score.


```python
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
```


```python
def nn_model(input_dim):
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim))
    model.add(Activation('tanh'))

    model.add(Dense(6))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    return model
```


```python
nn = nn_model(len(features))
nn.fit(training_data[features].values, np_utils.to_categorical(training_data.Class.values), verbose=1, epochs=10, batch_size=256)
```

    Epoch 1/10
    469/469 [==============================] - 3s 2ms/step - loss: 1.7330
    Epoch 2/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.4043
    Epoch 3/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.3516
    Epoch 4/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.3132
    Epoch 5/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.3103
    Epoch 6/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.2856
    Epoch 7/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.2851
    Epoch 8/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.2830
    Epoch 9/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.2703
    Epoch 10/10
    469/469 [==============================] - 1s 2ms/step - loss: 1.2549





    <tensorflow.python.keras.callbacks.History at 0x7fbd9672efd0>



### Log loss on the cross validation sample


```python
# predict each track
proba_nn = nn.predict_proba(validation_data[features].values)
```

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/sequential.py:425: UserWarning: `model.predict_proba()` is deprecated and will be removed after 2021-01-01. Please use `model.predict()` instead.
      warnings.warn('`model.predict_proba()` is deprecated and '



```python
log_loss(validation_data.Class.values, proba_nn)
```




    1.2588551006467399



# Quality metrics

Plot ROC curves and signal efficiency dependece from particle mometum and transverse momentum values.


```python
proba = proba_gb
```


```python
utils.plot_roc_curves(proba, validation_data.Class.values)
```


![png](Particle_identification_files/Particle_identification_37_0.png)



```python
utils.plot_signal_efficiency_on_p(proba, validation_data.Class.values, validation_data.TrackP.values, 60, 50)
plt.show()
```


![png](Particle_identification_files/Particle_identification_38_0.png)



```python
utils.plot_signal_efficiency_on_pt(proba, validation_data.Class.values, validation_data.TrackPt.values, 60, 50)
plt.show()
```


![png](Particle_identification_files/Particle_identification_39_0.png)

