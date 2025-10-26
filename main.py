import pandas as pd
from IPython.display import display

df = pd.read_json("arrFilteredDevices.json")
df[["strName", 'intReturnRadius', 'blnAutoPoints', "strDescription"]].sort_values("strName")