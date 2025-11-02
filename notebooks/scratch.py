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
# %%

import torch
import torch.nn.functional as F
dS = torch.tensor([-100., 1, 100])
C = torch.tensor([-2., -1., 0., 1., 2.])
for ds in dS:
    print(f"ds={ds: .1f}")
    for c in C:
        print(  f"C={c: .1f}: ", end='' )
        print(f"F.softplus(C*dS)={F.softplus(c*ds): .1f}, C*F.softplus(dS)={c * F.softplus(ds): .1f}, ")
        # print( f"add: {ds + c: .1f}, mult: {c * (1 + ds): .1f}, " , end='' )
        # print( f"softplus add: {F.softplus(ds + c): .1f}, softplus mult: {F.softplus(c * (1 + ds)): .1f}" )
    print()
# C * F.softplus(dS), F.softplus(C*dS)
