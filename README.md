# Events Temporal Up-sampling from H-FPS Brightness Changes Estimation

## Install
```
conda create -n ETUP
conda activate ETUP
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install dv, opence-python, numpy
```
You can change the third line to match your CUDA version from [here](https://pytorch.org/get-started/locally/). Please make sure PyTorch >= 1.0.


## Run
1. Download the [E2BC](https://drive.google.com/file/d/1Ut-42xrJ38G55hQ52qrmSLxYkc_KwBFF/view?usp=share_link) model and move it to model folder
2. Download an example file [Doraemon.aedat4](https://drive.google.com/file/d/1Z0Iu9ZR8eq_8Sn9XoKeIgtFKonQuwWSQ/view?usp=share_link) and move it to data folder
3. Up-sampling events:
```
python main.py
```


## Dataset: T-UP

[Download link](https://drive.google.com/drive/folders/1l1LL6GvdxdaOD-OBfGpoWIUygPwthjX6?usp=share_link)

<table>
    <thead>
        <tr>
            <th>Static scenes</th>
            <th>Sparse events</th>
            <th>Dense events</th>
            <th>APS frames</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>building</td>
            <td>9,081,952</td>
            <td>17,248,684</td>
            <td>247</td>
        </tr>
        <tr>
            <td>street</td>
            <td>8,149,959</td>
            <td>11,973,317</td>
            <td>351</td>
        </tr>
        <tr>
            <td>gym</td>
            <td>11,497,747</td>
            <td>25,331,865</td>
            <td>367</td>
        </tr>
        <tr>
            <td>table</td>
            <td>13,291,067</td>
            <td>31,907,991</td>
            <td>339</td>
        </tr>
        <tr>
            <td>calendar</td>
            <td>4,456,457</td>
            <td>13,353,352</td>
            <td>134</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th>Dynamic scenes</th>
            <th>Sparse events</th>
            <th>Dense events</th>
            <th>APS frames</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>parabola</td>
            <td>1,534,473</td>
            <td>5,634,897</td>
            <td>97</td>
        </tr>
        <tr>
            <td>roll</td>
            <td>1,907,543</td>
            <td>5,171,034</td>
            <td>90</td>
        </tr>
        <tr>
            <td>box flip</td>
            <td>3,402,791</td>
            <td>11,733,368</td>
            <td>213</td>
        </tr>
        <tr>
            <td>clockwise</td>
            <td>3,384,139</td>
            <td>12,663,820</td>
            <td>104</td>
        </tr>
        <tr>
            <td>gears</td>
            <td>695,962</td>
            <td>1,784,112</td>
            <td>59</td>
        </tr>
    </tbody>
</table>

Read data:


Install dv:
```
pip install dv
```

Read events and APS:
```python
import numpy as np
from dv import AedatFile

with AedatFile(file_path) as f:
    sparse_events = np.hstack([packet for packet in f['events'].numpy()])
    dense_events = np.hstack([packet for packet in f['events_1'].numpy()])
    aps = [packet for packet in f['frames']]


```
