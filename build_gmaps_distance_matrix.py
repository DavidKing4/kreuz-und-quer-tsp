import pandas as pd
import requests
import json

df = pd.read_json("arrFilteredDevices.json")
device_class_df = pd.json_normalize(df["objDeviceClass"]).add_prefix("device_class_")
merged_df = pd.merge(df, device_class_df, left_index=True, right_index=True)
filtered_df = merged_df[merged_df["device_class_strName"]=="Streetpoint"]

class GmapsDirectionGetter:

    def __init__(self, api_key):
        self.key = api_key

    def get_directions(self, lon1: float, lat1: float, lon2: float, lat2: float):
        payload  = {
            "origin":{
                "location":{
                    "latLng":{
                        "latitude": lat1,
                        "longitude": lon1
                    }
                }
            },
            "destination":{
                "location":{
                    "latLng":{
                        "latitude": lat2,
                        "longitude": lon2
                    }
                }
            },
            "travelMode": "WALK",
            "units": "METRIC"
        }
        
        header = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters"
        }
        response = requests.post(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            data=json.dumps(payload),
            headers=header
        )

        return response.json()["routes"][0]



gmaps_api_key = input("google maps api key: ")
gmaps = GmapsDirectionGetter(gmaps_api_key)

distance_matrix_meters = []
distance_matrix_seconds = []
for i, x in filtered_df.iterrows():
    current_distances_meters = []
    current_distances_seconds = []
    for j, y in filtered_df.iterrows():
        distance_meters = None
        distance_seconds = None
        if (i != j) and (i < j):
            gmaps_result = gmaps.get_directions(
                x["dblLongitude"], 
                x["dblLatitude"], 
                y["dblLongitude"],
                y["dblLatitude"], 
            )
            print(gmaps_result)
            distance_meters = gmaps_result["distanceMeters"]
            distance_seconds = int(gmaps_result["duration"][:-1])
        current_distances_meters.append(distance_meters)
        current_distances_seconds.append(distance_seconds)
    distance_matrix_meters.append(current_distances_meters)
    distance_matrix_seconds.append(current_distances_seconds)

def fix_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(i):
            matrix[i][j] = matrix[j][i]
    return(matrix)

fix_matrix(distance_matrix_seconds)
fix_matrix(distance_matrix_meters)

with open('gmap_distance_matrix_meters.txt', 'w') as f:
    for line in distance_matrix_meters:
        line_as_text = [str(x) if x is not None else '' for x in line ]
        f.write(f"{','.join(line_as_text)}\n")

with open('gmap_distance_matrix_seconds.txt', 'w') as f:
    for line in distance_matrix_seconds:
        line_as_text = [str(x) if x is not None else '' for x in line ]
        f.write(f"{','.join(line_as_text)}\n")
