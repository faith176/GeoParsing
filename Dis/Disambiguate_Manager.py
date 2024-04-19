import pickle
import re
from geopy.distance import geodesic
import numpy as np
import pandas as pd
import folium
from alive_progress import alive_bar
from sklearn.metrics import mean_absolute_error


class Disambiguate_Manager():
    def __init__(self, gaz, preprocess) -> None:
        self.gaz = gaz
        self.preprocess = preprocess
        self.default_location = {
            "name": "Unknown Location",
            "lat": 0.0,
            "lon": 0.0
        }
        self.relevant_fcodes = ["PPL", "PCLI", "ADM1", "ADM2", "PPLA", "PPLC", "PCLS", "ADMD", "STM", "AREA"]
        self.demonyms = pd.DataFrame()
        with open('data/geo_data/demonyms.pkl','rb') as f:
            self.demonyms = pickle.load(f)
        self.all_pop = pd.DataFrame()
        with open('data/geo_data/all_populations.pkl','rb') as f:
            self.all_pop = pickle.load(f)
            
            
    # Disambiguate New Input
    def map_locations(self, text, locations_list):
        locations = self.disambiguate(text,locations_list)
        print("-"*50)
        print(locations)
        print("-"*50)
        geoname_coordinates = {}
        for final in locations:
            name = final[0]
            lat_lon = final[1]
            if name.startswith("Location Unknown") == False:
                geoname_coordinates[name.title()] = lat_lon
        if len(geoname_coordinates) == 0:
            map = folium.Map(location=[0, 0], zoom_start=2, lang='en')
        else:
            avg_lat = sum(coord[0] for coord in geoname_coordinates.values()) / len(geoname_coordinates.values())
            avg_lon = sum(coord[1] for coord in geoname_coordinates.values()) / len(geoname_coordinates.values())
            map = folium.Map(location=[avg_lat, avg_lon], zoom_start=4, lang='en')
            for name, coordinates in geoname_coordinates.items():
                popup_content = f"<b>{name}</b><br>Coordinates: {coordinates[0]}, {coordinates[1]}"
                folium.Marker(
                    coordinates, 
                    popup=popup_content,
                    auto_open=True,
                    ).add_to(map)
        title_html = '<h3 align="center" style="font-size:16px"><b>Locations</b></h3>'
        map.get_root().html.add_child(folium.Element(title_html))
        return map
            
            
    def disambiguate_corpus(self, type="main", distance_threshold = 60):
        correct_count = 0
        true_labels = []
        predicted_labels = []
        with alive_bar(len(self.preprocess.corpus), force_tty=True) as bar:
            for book in self.preprocess.corpus:
                locs = []
                for loc in book["toponyms"]:
                    if loc.get("geonameid") is not None and loc.get("fcode") in self.relevant_fcodes:
                        if len(self.all_pop[self.all_pop["geonameid"] == loc.get("geonameid")]["population"]) == 1 and self.all_pop[self.all_pop["geonameid"] == loc.get("geonameid")]["population"].values[0] >= 1000:
                            locs.append(loc)
                loc_list = [loc["phrase"].lower() for loc in locs]
                if type == "population":
                    final_predictions = self.disambiguate_baseline_population(book["text"],loc_list)
                elif type == "distance":
                    final_predictions = self.disambiguate_baseline_distance(book["text"],loc_list)
                else:
                    final_predictions = self.disambiguate(book["text"],loc_list)
                for item in locs:
                    for prediction in final_predictions:
                        if item.get("start") == str(prediction[-1]):
                            true_labels.append((item["lat"], item["lon"]))
                            predicted_labels.append((prediction[1][0], prediction[1][1]))
                            
                            distance_apart = self.get_distance(prediction[1][0], prediction[1][1], item["lat"], item["lon"])
                            if int(item.get("geonameid")) == int(prediction[2]) or distance_apart <= distance_threshold:
                                correct_count += 1
                bar()
        accuracy = (correct_count / len(true_labels)) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        mse = self.mse(true_labels, predicted_labels)
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(true_labels, predicted_labels):.2f}")
        return true_labels, predicted_labels
        
    # Scoring Functions------------------------------------------------------
    def mse(self, true, predicted):
        mse = 0
        for true_coord, pred_coord in zip(true, predicted):
            true_lat, true_lon = true_coord
            pred_lat, pred_lon = pred_coord
            # Calculate squared error for latitude and longitude
            error_lat = (true_lat - pred_lat) ** 2
            error_lon = (true_lon - pred_lon) ** 2
            # Calculate the MSE
            mse += error_lat + error_lon
        mse /= len(true)
        return mse
    
    
    # Disambiguation Functions------------------------------------------------------
    def disambiguate_baseline_population(self, text, loc_list):
        candidates = {index: [] for index, _ in enumerate(loc_list)}
        loc_indices,  full_loc_list, final_disambiguated = self.map_locations_to_indices(text, loc_list, True)

        # Find all guaranteed locations, pick highest population for locations with multiple candidates, return unknown for those without a match
        for loc_index in full_loc_list:
            loc = loc_indices[loc_index][0]
            can = self.gaz.get_location_candidates(loc, 0) # candidate are returned from highest population to least
            if len(can) == 0:
                new_cands = self.gaz.get_location_candidates(loc, 1)
                if len(new_cands) > 0:
                    match = new_cands[0]
                    candidates[loc_index] = new_cands
                    final_disambiguated[loc_index] = ((match["name"]), match["lat_lon"], match["geonamesid"], loc_indices[loc_index][1])
                else:
                    candidates[loc_index] = []
                    final_disambiguated[loc_index] = (f"Location Unknown: {loc.title()}", (self.default_location["lat"], self.default_location["lon"]), -1)
            else:
                match = can[0]
                candidates[loc_index] = can
                final_disambiguated[loc_index] = ((match["name"]), match["lat_lon"], match["geonamesid"], loc_indices[loc_index][1])                
        return self.process_final(final_disambiguated)
    
    
    def disambiguate_baseline_distance(self, text, loc_list):
        partial_loc_list = []
        candidates = {index: [] for index, _ in enumerate(loc_list)}
        loc_indices,  full_loc_list, final_disambiguated = self.map_locations_to_indices(text, loc_list, False)
        
        # First Iteration - find all guaranteed locations
        for loc_index in full_loc_list:
            loc = loc_indices[loc_index][0]
            can = self.gaz.get_location_candidates(loc, 0)
            if len(can) == 1:
                match = can[0]
                candidates[loc_index] = can
                final_disambiguated[loc_index] = ((match["name"]), match["lat_lon"], match["geonamesid"], loc_indices[loc_index][1])
            else:
                partial_loc_list.append(loc_index)
                candidates[loc_index] = can
                
        
        temp_final_disambiguated = final_disambiguated
        for loc_index in partial_loc_list:
            loc = loc_indices[loc_index][0]
            og_cands = candidates[loc_index]
            if len(og_cands) > 0:
                # Returns candidate that is the closest distance-wise to the other guaranteed locations
                nearby = [coords[1] for coords in temp_final_disambiguated.values() if coords[-1] != -1]
                check = [c["lat_lon"] for c in og_cands]
                index, _ = self.closest_coordinate_to_all(nearby, check)
                match = og_cands[index]
                candidates[loc_index] = og_cands
                final_disambiguated[loc_index] = ((match["name"]), match["lat_lon"], match["geonamesid"], loc_indices[loc_index][1])
            else:
                new_cands = self.gaz.get_location_candidates(loc, 1)
                if len(new_cands) > 0:
                    nearby = [coords[1] for coords in temp_final_disambiguated.values() if coords[-1] != -1]
                    check = [c["lat_lon"] for c in new_cands]
                    index, _ = self.closest_coordinate_to_all(nearby, check)
                    match = new_cands[index]
                    candidates[loc_index] = new_cands
                    final_disambiguated[loc_index] = ((match["name"]), match["lat_lon"], match["geonamesid"], loc_indices[loc_index][1])
                    
                else:
                    final_disambiguated[loc_index] = (f"Location Unknown: {loc.title()}", (self.default_location["lat"], self.default_location["lon"]), -1)
                  
        return self.process_final(final_disambiguated)
    
        
        
    # Combined disambiguation
    def disambiguate(self, text, loc_list):
        partial_loc_list = []
        candidates = {index: [] for index, _ in enumerate(loc_list)}
        geographic_context = []
        loc_indices,  full_loc_list, final_disambiguated = self.map_locations_to_indices(text, loc_list, False)
        
        # First Iteration - find all guaranteed locations
        for loc_index in full_loc_list:
            loc = loc_indices[loc_index][0]
            can = self.gaz.get_location_candidates(loc, 0)
            if len(can) == 1:
                match = can[0]
                geographic_context.append(match["country_code"])
                candidates[loc_index] = can
                final_disambiguated[loc_index] = ((match["name"]), match["lat_lon"], match["geonamesid"], loc_indices[loc_index][1])
            elif len(can) > 1:
                partial_loc_list.append(loc_index)
                candidates[loc_index] = can
            else:
                expanded_candidates = self.gaz.get_location_candidates(loc, 1)
                candidates[loc_index] = expanded_candidates
                if len(expanded_candidates) > 0 :
                    partial_loc_list.append(loc_index)
                else:
                    # If no location is matched after expansion
                    final_disambiguated[loc_index] = (f"Location Unknown: {loc.title()}", (self.default_location["lat"], self.default_location["lon"]), -1)
            
        # Assign population distance scoring and pick best candidate
        for loc_index in partial_loc_list:
            loc = loc_indices[loc_index][0]
            cands = candidates[loc_index]
            if len(cands) == 1:
                match = cands[0]
                geographic_context.append(match["country_code"])
                candidates[loc_index] = cands
                final_disambiguated[loc_index] = ((match["name"]), match["lat_lon"], match["geonamesid"] , loc_indices[loc_index][1])
            else:
                best_candidate = self.check_local_context(cands, final_disambiguated)
                match = best_candidate
                candidates[loc_index] = cands
                final_disambiguated[loc_index] = ((match["name"]), match["lat_lon"], match["geonamesid"] , loc_indices[loc_index][1])
        return self.process_final(final_disambiguated)
    
    
    
    # Helper Functions ------------------------------------------------------------------------------
    def get_distance(self, lat1, lon1, lat2, lon2):
        coords1 = (lat1, lon1)
        coords2 = (lat2, lon2)
        distance = geodesic(coords1, coords2).kilometers
        return distance
    
    def check_local_context(self, curr_candidates, final_disambiguated, distance_weight=0.6):
        distances = []
        populations = []
        final_disambiguated_coordinates = [coords[1] for coords in final_disambiguated.values() if coords[-1] != -1]
        dis_km = []
        
        for curr_index in range(len(curr_candidates)):
            curr_lat_lon = [curr_candidates[curr_index]["lat_lon"]]
            _ , min_distance = self.closest_coordinate_to_all(final_disambiguated_coordinates, curr_lat_lon)
            if min_distance == 0:  # Handle division by zero
                distances.append(float('inf'))  # Set a very large value as distance
            else:
                distances.append(1 / min_distance)  # Inverse since small distances should increase the score
            populations.append(curr_candidates[curr_index]["population"])
            dis_km.append(min_distance)
        # Normalize populations and distances
        normalized_populations = np.array(populations) / max(populations)
        normalized_distances = np.array(distances) / max(distances)
        
        combined_rankings = (1 - distance_weight) * normalized_populations + distance_weight * normalized_distances
        max_score_index =  np.argmax(combined_rankings)
        return curr_candidates[max_score_index]
    
    
    def closest_coordinate_to_all(self, nearby, check):
        if len(nearby) == 0:
            return 0, float('inf')
        avg_coord = np.mean(nearby, axis=0)
        min_distance = float('inf')
        closest_coord_index = 0 # Returns highest population by default if there are no guaranteed coordinates so far
        
        for index, check_coord in enumerate(check):
            distance = geodesic(avg_coord, check_coord).kilometers
            if distance < min_distance:
                min_distance = distance
                closest_coord_index = index
        return closest_coord_index, min_distance

    def map_locations_to_indices(self,text, initial_loc_list, baseline=False):
        indices = {}
        valid_locs = []
        found = []
        final_disambiguated = {}
        id_count = 0
        for location in initial_loc_list:
            if location not in found:
                instances = re.finditer(re.escape(location.lower()), text.lower())
                instances_check = 0
                for m in instances:
                    instances_check += 1
                    break
                if instances_check == 1:
                    for match in re.finditer(re.escape(location.lower()), text.lower()):
                        if baseline == False and location in self.demonyms['Phrase'].unique():
                            filtered_row = self.demonyms.loc[self.demonyms['Phrase'] == location]
                            found.append(location)
                            indices[id_count] = (filtered_row['Name'].values[0], match.start())
                        else:
                            found.append(location)
                            indices[id_count] = (location, match.start())
                        valid_locs.append(id_count)
                        id_count += 1
                else:
                    final_disambiguated[id_count] = (f"Location Unknown: {location}", (self.default_location["lat"], self.default_location["lon"]), -1)
                    indices[id_count] = (location, -1)
                    id_count += 1
            else:
                pass
        return indices, valid_locs, final_disambiguated
        

    def process_final(self,final_disambiguated ):
        sorted_items = sorted(final_disambiguated.items(), key=lambda item: item[1][-1])
        return [value for _, value in sorted_items]