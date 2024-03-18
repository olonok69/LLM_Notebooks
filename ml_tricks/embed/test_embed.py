## The embed() function of IPython module makes it possible to embed IPython in
## your Python codes’ namespace.Thereby you can leverage IPython features like
## object introspection and tab completion, in default Python environment.


import pandas as pd
from IPython import embed


def tolower(x):
    return x.lower()


data = pd.read_csv("timetables.csv")
embed()
data2 = data.query("masterheadcode == '1P45'")
print(len(data2), len(data))
