import math
import pandas as pd
from IPython.display import display


# df[["strName", 'intReturnRadius', 'blnAutoPoints', "strDescription"]].sort_values("strName")
df = pd.read_json("arrFilteredDevices.json")
device_class_df = pd.json_normalize(df["objDeviceClass"]).add_prefix("device_class_")
merged_df = pd.merge(df, device_class_df, left_index=True, right_index=True)
filtered_df = merged_df[merged_df["device_class_strName"]=="Streetpoint"]

def get_crow_distince(x1: long, y1: long, x2: long, y2: long) -> long:
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

distance_matrix = []
for i, x in filtered_df.iterrows():
    current_distances = []
    for j, y in filtered_df.iterrows():
        distance = None
        if i != j:
            distance = get_distince(
                x["dblLatitude"], 
                x["dblLongitude"], 
                y["dblLatitude"], 
                y["dblLongitude"]
            )
        current_distances.append(distance)
    distance_matrix.append(current_distances)



# 64_424_509_440
# dblLatitude
# dblLongitude
