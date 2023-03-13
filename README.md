# Events Temporal Up-sampling from H-FPS Brightness Changes Estimation

## Dataset: T-UP 

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

```
pip install dv
```


```python
import numpy as np
from dv import AedatFile

    with AedatFile(file_path) as f:
        sparse_events = np.hstack([packet for packet in f['events'].numpy()])
        dense_events = np.hstack([packet for packet in f['events_1'].numpy()])
        aps = [packet for packet in f['frames']]


```
