import copy
import datetime
import math
import pandas as pd
import numpy as np
from itertools import combinations

# from python_tsp.exact import solve_tsp_branch_and_bound
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
from numbers import Number


# df[["strName", 'intReturnRadius', 'blnAutoPoints', "strDescription"]].sort_values("strName")
df = pd.read_json("arrFilteredDevices.json")
device_class_df = pd.json_normalize(df["objDeviceClass"]).add_prefix("device_class_")
merged_df = pd.merge(df, device_class_df, left_index=True, right_index=True)
filtered_df = merged_df[merged_df["device_class_strName"] == "Streetpoint"]

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

    R = 6371  # Radius of the earth in km

    φ1 = x1 * math.pi / 180
    φ2 = x2 * math.pi / 180
    Δφ = (x2 - x1) * math.pi / 180
    Δλ = (y2 - y1) * math.pi / 180

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
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

initial_distance_matrix = []
with open("gmap_distance_matrix_meters.txt", "r") as f:
    for line in f.readlines():
        initial_distance_matrix.append(
            [int(x) if x != "" else np.inf for x in line[:-1].split(",")]
        )

d = [
    [np.inf, 1195, 0, 44, 99],
    [1195, np.inf, 767, 0, 120],
    [0, 767, np.inf, 233, 411],
    [44, 0, 233, np.inf, 0],
    [99, 120, 411, 0, np.inf],
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

    for s in range(2, n - 1):
        print("current no. nodes: ", s)
        print("current size of subsets: ", len(subsets))
        level_subsets = {}
        level_paths = {}
        for S in combinations(range(1, n), s):
            distance_min = None
            path_min = None
            for k in range(s):
                no_k = S[:k] + S[k + 1 :]
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
    for k in range(n - 1):
        no_k = S[:k] + S[k + 1 :]
        print(no_k)
        distance = subsets[(no_k)] + d[paths[(no_k)][-1]][S[k]] + d[S[k]][0]
        print(distance)
        if (distance_final is None) or (distance < distance_final):
            distance_final = distance
            path_final = paths[no_k] + (S[k],) + (0,)
            print("new_best: ", distance_final, path_final)

    return distance_final, path_final


# increase performance with ordered list of nodes?
def get_minimum_bound(
    distance_matrix: list[list[Number]],
    mask_matrix: list[list[Number]] = None,
    starting_node: int = None,
    current_node: int = None,
    traveled: int = 0,
) -> float:
    minimum_bound = 0
    for i, (node, node_mask) in enumerate(zip(distance_matrix, mask_matrix)):
        if any(node_mask):
            s1, s2 = np.inf, np.inf
            for d, m in zip(node, node_mask):
                if m:
                    if d < s2:
                        if d < s1:
                            s2 = s1
                            s1 = d
                        else:
                            s2 = d
            if i == current_node:
                minimum_bound += s1 / 2
            else:
                minimum_bound += (s1 + s2) / 2

    if starting_node:
        s1 = np.inf
        for i in range(len(distance_matrix)):
            if mask_matrix[i][starting_node]:
                if distance_matrix[i][starting_node] < s1:
                    s1 = distance_matrix[i][starting_node]
        minimum_bound += s1 / 2

    return minimum_bound + traveled


# def branch_and_bound_rec():
#     pass


initial_mask_matrix = [
    [True if i != j else False for i in range(len(initial_distance_matrix[0]))]
    for j in range(len(initial_distance_matrix))
]


def branch_and_bound(
    distance_matrix: list[list[Number]],
    mask_matrix: list[list[bool]],
    starting_node: int,
    current_node: int,
    traveled: int,
    visited: list[int],
    best_distance: Number,
    best_path: list[int],
    rec_level: int,
    logging: bool = False,
) -> tuple(Number, list[int]):

    # base case
    if len(visited) == len(distance_matrix) - 2:
        # take shortest of two remaining paths

        # print(current_node, starting_node)
        # for line in mask_matrix:
        #     print("".join(["1" if x else "_" for x in line]))

        unvisited1 = None
        unvisited2 = None
        for i, mask in enumerate(mask_matrix[current_node]):
            # print(mask, i, starting_node)
            if mask and (i != starting_node):
                # print("eep")
                if unvisited1 is None:
                    unvisited1 = i
                else:
                    unvisited2 = i

        # print(unvisited1, unvisited2)
        # print()

        # unvisited1 = mask_matrix[current_node][1:].index(True) + 1
        # unvisited2 = (
        #     mask_matrix[current_node][unvisited1 + 1 :].index(True) + unvisited1 + 1
        # )

        d1 = (
            distance_matrix[current_node][unvisited1]
            + distance_matrix[unvisited1][unvisited2]
            + distance_matrix[unvisited2][starting_node]
        )
        d2 = (
            distance_matrix[current_node][unvisited2]
            + distance_matrix[unvisited2][unvisited1]
            + distance_matrix[unvisited1][starting_node]
        )
        if not best_distance or (traveled + min(d1, d2)) < best_distance:
            best_path = visited + (
                [unvisited1, unvisited2, starting_node]
                if d1 <= d2
                else [unvisited2, unvisited1, starting_node]
            )
            print(datetime.datetime.now())
            print(
                "new best: ",
                mask_matrix[current_node],
                unvisited1,
                unvisited2,
                starting_node,
            )
            print("new best distance: ", (traveled + min(d1, d2)), best_path)

            if logging:
                with open("best_found_log.txt", "a") as f:
                    f.write(f"{datetime.datetime.now()}\n")
                    f.write(f"new best: {best_path}\n")
                    f.write(f"new best distance: {(traveled + min(d1, d2))}\n")

            return ((traveled + min(d1, d2)), best_path)
        else:
            return (-1, [])

    if current_node == starting_node:
        minimum_bound = get_minimum_bound(
            distance_matrix,
            mask_matrix,
            traveled=traveled,
        )
    else:
        minimum_bound = get_minimum_bound(
            distance_matrix,
            mask_matrix,
            None,
            current_node,
            traveled=traveled,
        )

    # if visited in [[10, 7, 9, 1, 8, 2, 6, 0, 3, 5, 4][:x] for x in range(5, 9)]:
    #     print(visited)
    #     print(minimum_bound, traveled)
    #     print(mask_matrix)
    # for line in mask_matrix:
    #     print("".join(["1" if x else "_" for x in line]))

    if best_distance and (minimum_bound >= best_distance):
        # print(minimum_bound, best_distance)
        return (-1, [])

    # use gready algorithm start at node 0 and go to closest node
    # queue of closest nodes
    for i in np.argsort(distance_matrix[current_node]):
        # 1: update mask
        # if rec_level == 1:
        #     print(i)
        # if i == 3:
        #     for node in mask_matrix:
        #         for x in node:
        #             print(1, end="") if x else print(0, end="")
        #         print()

        if mask_matrix[current_node][i]:
            new_mask = copy.deepcopy(mask_matrix)
            for j in range(len(new_mask)):
                new_mask[current_node][j] = False
                new_mask[j][i] = False

            new_mask[i][starting_node] = False

            new_best_distance, new_best_path = branch_and_bound(
                distance_matrix,
                new_mask,
                starting_node,
                current_node=i,
                traveled=traveled + distance_matrix[current_node][i],
                visited=visited + [int(i)],
                best_distance=best_distance,
                best_path=best_path,
                rec_level=rec_level + 1,
                logging=logging,
            )
            # mask_matrix[i][current_node] = False ?

            if new_best_distance != -1:
                best_distance = new_best_distance
                best_path = new_best_path

    return (best_distance, best_path)


# best_dist = 1000000000000000
# best = None
# for i in range(100):
#     # print(i)
#     a = solve_tsp_local_search(np.array(distance_matrix))
#     if a[1] < best_dist:
#         print("new best by local_search: ", a)
#         best = a
#         best_dist = a[1]

#     a = solve_tsp_simulated_annealing(np.array(distance_matrix))
#     if a[1] < best_dist:
#         print("new best by simulated_annealing: ", a)
#         best = a
#         best_dist = a[1]

# # a = (solve_tsp_local_search(np.array(distance_matrix)))
# # b = (solve_tsp_simulated_annealing(np.array(distance_matrix)))
# # print(a)
# # print(b)

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
            icon=folium.Icon(color="blue", icon="flag"),
        ).add_to(m)

    # Draw route as polyline
    folium.PolyLine(points, weight=4, opacity=0.8).add_to(m)

    return m


# # Example usage
reset_index_df = filtered_df.reset_index()
# # a_df = reset_index_df.iloc[a[0]]
# # b_df = reset_index_df.iloc[b[0]]
# best_df = reset_index_df.iloc[best[0]]

# # ma = plot_tsp_path(a_df, "dblLatitude", "dblLongitude")
# # ma.save("tsp_route_a.html")

# # mb = plot_tsp_path(b_df, "dblLatitude", "dblLongitude")
# # mb.save("tsp_route_b.html")

# mb = plot_tsp_path(best_df, "dblLatitude", "dblLongitude")
# mb.save("tsp_route_best.html")
n = 11
subset_matrix = [x[:n] for x in initial_distance_matrix][:n]
subset_mask = [
    [True if i != j else False for i in range(len(subset_matrix[0]))]
    for j in range(len(subset_matrix))
]
# a = branch_and_bound(
#     subset_matrix,
#     subset_mask,
#     starting_node=1,
#     current_node=1,
#     traveled=0,
#     visited=[1],
#     best_distance=None,
#     best_path=None,
#     rec_level=1,
#     logging=True,
# )
# d_mask = [[True if i != j else False for i in range(len(d[0]))] for j in range(len(d))]
# subset_matrix = [[y - min(x) for y in x] for x in subset_matrix]
# for i in range(len(subset_matrix)):
#     print(f"starting at node {i}")
#     a = branch_and_bound(
#         copy.deepcopy(subset_matrix),
#         copy.deepcopy(subset_mask),
#         starting_node=i,
#         current_node=i,
#         traveled=0,
#         visited=[i],
#         best_distance=None,
#         best_path=None,
#         rec_level=1,
#         # logging=True,
#     )
#     print(a)
#     a_df = reset_index_df.iloc[a[1]]
#     ma = plot_tsp_path(a_df, "dblLatitude", "dblLongitude")
#     ma.save(f"tsp_route_a{i}.html")

# b = tsp(subset_matrix)

full = branch_and_bound(
    initial_distance_matrix,
    initial_mask_matrix,
    starting_node=9,
    current_node=9,
    traveled=0,
    visited=[9],
    best_distance=None,
    best_path=None,
    rec_level=1,
    logging=True,
)

# Example usage
reset_index_df = filtered_df.reset_index()
best_df = reset_index_df.iloc[full[1]]

# print(solve_tsp_simulated_annealing(np.array(subset_matrix)))


# b_df = reset_index_df.iloc[list(b[1])]
# mb = plot_tsp_path(b_df, "dblLatitude", "dblLongitude")
# mb.save("tsp_route_b.html")

mb = plot_tsp_path(best_df, "dblLatitude", "dblLongitude")
mb.save("tsp_route_best.html")
