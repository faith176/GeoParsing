import os
from .BKTree import BKTree
# from BKTree import BKTree
import pandas as pd
from alive_progress import alive_bar
import pickle

class Gazetteer:
    def __init__(self, locationFilePath="./data/geo_data/specific_geonames_locations.pkl") -> None:
        self.columns = ["ID", "Name", "Country_Code", "Population", "Latitude", "Longitude",  "FeatureCode", "AltCodes", "Children", "Parent", "Nationality", "Alt"]
        self.gaz_df = pd.read_pickle(locationFilePath)
        self.gaz_df.columns = self.columns
        self.locationsArray = []
        self.locationsList = {}
        self.processLocationsList()
        self.BKTree = BKTree([(element["name"], element["geonamesid"]) for element in self.locationsArray])
            
    def processLocationsList(self):
        if os.path.exists('data/saved_data/Gaz/locations_list.pkl'):
            print("Retrieving Locations Array from Saved Data")
            self.retrieve_loc_array()
            print(f"Corpus has {len(self.locationsList)} Locations")
        else:
            print("Creating Gazetteer")
            with alive_bar(len(self.gaz_df), force_tty=True) as bar:
                for _, row in self.gaz_df.iterrows():
                    metadata = {
                        'geonamesid': row["ID"],
                        'name': row["Name"].lower(),
                        'country_code': row["Country_Code"],
                        'population': row["Population"],
                        'lat_lon': (row["Latitude"], row["Longitude"]),
                        'feature_codes': row["FeatureCode"],
                        'alt_codes': row["AltCodes"],
                        'children': row["Children"],
                        'parent': row["Parent"],
                    }
                    if row["Nationality"] is not None:
                        for nat in row["Nationality"].split(", "):
                            self.locationsArray.append({
                                'geonamesid': row["ID"],
                                'name': nat.lower(),
                    })
                    if row["Alt"] is not None:
                        for a in row["Alt"].split(","):
                            self.locationsArray.append({
                                'geonamesid': row["ID"],
                                'name': a.lower(),
                    })
                    if row["FeatureCode"] == "PCLI" and type(row["Country_Code"]) == str:
                        self.locationsArray.append({
                                'geonamesid': row["ID"],
                                'name': row["Country_Code"].lower(),
                        })
                        
                    self.locationsArray.append(metadata)
                    self.locationsList[row["ID"]] = metadata
                    bar()
            self.save_loc_array()
                
    def save_loc_array(self):            
        with open('data/saved_data/Gaz/locations_list.pkl','wb') as f:
            pickle.dump(self.locationsList, f)
            print("Saved Data")
            
    def retrieve_loc_array(self):            
        with open('data/saved_data/Gaz/locations_list.pkl','rb') as f:
            loaded_l = pickle.load(f)
            self.locationsList = loaded_l
            
    def get_location_candidates(self, query, max_distance):
        return sorted([entry for entry in [self.locationsList[loc[3].geoID] for loc in self.BKTree.search(query, max_distance)]], key=lambda x: x.get("population", 0), reverse=True)
    
    def check_location_exist(self, query):
        return (len(self.BKTree.search(query, 0)) > 0)
