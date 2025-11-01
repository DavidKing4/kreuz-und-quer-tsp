import math
import pandas as pd
from IPython.display import display
from itertools import combinations


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
            distance = get_crow_distince(
                x["dblLatitude"], 
                x["dblLongitude"], 
                y["dblLatitude"], 
                y["dblLongitude"]
            )
        current_distances.append(distance)
    distance_matrix.append(current_distances)

# def tsp(distance_matrix: list[list[int]]) -> tuple[int, list[int]]:
d=[[0,1,2,3],[2,0,1,1],[2,2,0,1],[2,3,2,0]]

subsets = {}
paths = {}
n = len(d)
# 0 is our starting point instead of 1
for k in range(1, n):
    subsets[((k,), 0)] = d[0][k]
    paths[((k,), 0)] = (k, 0)
    print(k, d[0][k])
print(subsets)

for s in range(2, n+1):
    for S in combinations(range(1, n), s):
        print("S: ",S)
        distance_min = None
        k_min = None
        for k in range(s):
            print("k: ",k)
            no_k = S[:k] + S[k+1:]
            distance = subsets[(no_k, 0)] + d[no_k[-1]][S[k]]
            if (distance_min is None) or (distance < distance_min):
                distance_min = distance
                path_min = no_k + (S[k],)
                k_min = k
        subsets[(S, 0)] = distance_min
        paths[(S, 0)] = (0,) + path_min
        print(subsets)
        print(paths)
        

        
            
        # subsets[] = smallest
            


# 64_424_509_440
# dblLatitude
# dblLongitude
