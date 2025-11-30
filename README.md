**###Step one:** clone the path: https://github.com/lukewbauer/stock-portfolio-optimization
**##Step two:**

```python
**##import the requirements:**
 import sys
import os
 
# Install idaes-pse if running in Google Colab
if 'google.colab' in sys.modules:
    print('Installing idaes-pse...')
    !pip install idaes-pse --pre
    !idaes get-extensions --to ./bin
    os.environ['PATH'] += ':bin'
    print('idaes-pse installation complete.')
 
import numpy
import pandas
import matplotlib
import seaborn
import yfinance
import pyomo
import idaes.core as idaes_pse # Assuming idaes-pse is meant to be imported this way
!apt-get update
!apt-get install -y coinor-ipopt
from pyomo.environ import SolverFactory
solver = SolverFactory('ipopt')
%cd ..
```

**##Step Three**: **run this code**

os.chdir('/content/stock-portfolio-optimization')

!python main.py
