import math
import pandas as pd
import numpy as np
from itertools import combinations
from python_tsp.exact import solve_tsp_branch_and_bound
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing


# df[["strName", 'intReturnRadius', 'blnAutoPoints', "strDescription"]].sort_values("strName")
df = pd.read_json("arrFilteredDevices.json")
device_class_df = pd.json_normalize(df["objDeviceClass"]).add_prefix("device_class_")
merged_df = pd.merge(df, device_class_df, left_index=True, right_index=True)
filtered_df = merged_df[merged_df["device_class_strName"]=="Streetpoint"]

# function getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2) {
#   var R = 6371; // Radius of the earth in km
#   var dLat = deg2rad(lat2-lat1);  // deg2rad below
#   var dLon = deg2rad(lon2-lon1); 
#   var a = 
#     Math.sin(dLat/2) * Math.sin(dLat/2) +
#     Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * 
#     Math.sin(dLon/2) * Math.sin(dLon/2)
#     ; 
#   var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
#   var d = R * c; // Distance in km
#   return d;
# }

# function deg2rad(deg) {
#   return deg * (Math.PI/180)
# }

# code copied form js code
def get_crow_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    
    R = 6371 # Radius of the earth in km

    φ1 = x1 * math.pi/180
    φ2 = x2 * math.pi/180
    Δφ = (x2-x1) * math.pi/180
    Δλ = (y2-y1) * math.pi/180

    a = math.sin(Δφ/2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c
    # return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# distance_matrix = []
# for i, x in filtered_df.iterrows():
#     current_distances = []
#     for j, y in filtered_df.iterrows():
#         distance = None
#         if i != j:
#             distance = get_crow_distance(
#                 x["dblLongitude"], 
#                 x["dblLatitude"], 
#                 y["dblLongitude"],
#                 y["dblLatitude"], 
#             )
#         current_distances.append(distance)
#     distance_matrix.append(current_distances)

distance_matrix = []
with open('gmap_distance_matrix_seconds.txt', 'r') as f:
    for line in f.readlines():
        distance_matrix.append([int(x) if x != '' else None for x in line[:-1].split(",")])

d = [
    [0,  3,  4,  2,  7,  8,  5,  6,  4,  9],
    [3,  0,  2,  5,  6,  4,  7,  3,  8,  5],
    [4,  2,  0,  6,  3,  5,  4,  7,  6,  2],
    [2,  5,  6,  0,  4,  3,  5,  6,  7,  8],
    [7,  6,  3,  4,  0,  2,  6,  5,  4,  3],
    [8,  4,  5,  3,  2,  0,  4,  6,  5,  7],
    [5,  7,  4,  5,  6,  4,  0,  3,  2,  6],
    [6,  3,  7,  6,  5,  6,  3,  0,  4,  5],
    [4,  8,  6,  7,  4,  5,  2,  4,  0,  3],
    [9,  5,  2,  8,  3,  7,  6,  5,  3,  0]
]


# actually wrote this myself
def tsp(d: list[list[int]]) -> tuple[int, tuple[int]]:
    subsets = {}
    paths = {}
    n = len(d)
    # 0 is our starting point instead of 1
    for k in range(1, n):
        subsets[(k,)] = d[0][k]
        paths[(k,)] = (0, k)

    for s in range(2, n-1):
        print("current no. nodes: ", s)
        print("current size of subsets: ", len(subsets))
        level_subsets = {}
        level_paths = {}
        for S in combinations(range(1, n), s):
            distance_min = None
            path_min = None
            for k in range(s):
                no_k = S[:k] + S[k+1:]
                distance = subsets[(no_k)] + d[paths[(no_k)][-1]][S[k]]
                if (distance_min is None) or (distance < distance_min):
                    distance_min = distance
                    path_min = paths[no_k] + (S[k],)
            level_subsets[S] = distance_min
            level_paths[S] = path_min   
        subsets = level_subsets
        paths = level_paths


    S = tuple(range(1, n))
    distance_final = None
    path_final = None
    print(S)
    for k in range(n-1):
        no_k = S[:k] + S[k+1:]
        print(no_k)
        distance = subsets[(no_k)] +  d[paths[(no_k)][-1]][S[k]] +  d[S[k]][0]
        print(distance)
        if (distance_final is None) or (distance < distance_final):
            distance_final = distance
            path_final = paths[no_k] + (S[k],) + (0,)
            print("new_best: ", distance_final, path_final)

    return distance_final, path_final

best_dist = 1000000000000000
best = None
for i in range(100):
    # print(i)
    a = solve_tsp_local_search(np.array(distance_matrix))
    if a[1] < best_dist:
        print("new best by local_search: ", a)
        best = a
        best_dist = a[1]

    a = solve_tsp_simulated_annealing(np.array(distance_matrix))
    if a[1] < best_dist:
        print("new best by simulated_annealing: ", a)
        best = a
        best_dist = a[1]

# a = (solve_tsp_local_search(np.array(distance_matrix)))
# b = (solve_tsp_simulated_annealing(np.array(distance_matrix)))
# print(a)
# print(b)

# best = solve_tsp_branch_and_bound(np.array(distance_matrix))

# chat gpt poop
import folium

def plot_tsp_path(df, lat_col="latitude", lon_col="longitude"):
    # Create base map centered around the midpoint of data
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    points = []

    for _, row in df.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        points.append((lat, lon))
        
        popup_text = f"""
        Station ID: {row['strStationId']}<br>
        City: {row['strCity']}<br>
        Street: {row['strStreet']}
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=row["strStreet"],
            icon=folium.Icon(color="blue", icon="flag")
        ).add_to(m)

    # Draw route as polyline
    folium.PolyLine(points, weight=4, opacity=0.8).add_to(m)

    return m




# Example usage
reset_index_df = filtered_df.reset_index()
# a_df = reset_index_df.iloc[a[0]]
# b_df = reset_index_df.iloc[b[0]]
best_df = reset_index_df.iloc[best[0]]

# ma = plot_tsp_path(a_df, "dblLatitude", "dblLongitude")
# ma.save("tsp_route_a.html")

# mb = plot_tsp_path(b_df, "dblLatitude", "dblLongitude")
# mb.save("tsp_route_b.html")

mb = plot_tsp_path(best_df, "dblLatitude", "dblLongitude")
mb.save("tsp_route_best.html")
