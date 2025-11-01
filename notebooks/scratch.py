# %%
import numpy as np
import pandas as pd

coeff = np.linspace(-2, 2, 5)[:, None]
v = np.linspace(0.8, 1.2, 3)[None]
y = (v - 1.0) * coeff + 1.0
print(coeff, y)

# %%
df = pd.DataFrame(y, index=coeff.squeeze(), columns=v.squeeze())
df.index.name = 'coeff'
df.columns.name = 'v'
df
