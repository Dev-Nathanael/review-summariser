import requests
import json
from serpapi import GoogleSearch
import math
from typing import List, Dict, Tuple
import time

class PlaceFinder:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.user_lat = None
        self.user_lng = None
    
    def get_user_location(self, location_query: str) -> Tuple[float, float]:
        """
        Get latitude and longitude from location query using SerpAPI
        """
        params = {
            "engine": "google",
            "q": f"{location_query} coordinates",
            "api_key": self.serpapi_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Try to extract coordinates from search results
        if "answer_box" in results:
            answer = results["answer_box"]
            if "coordinates" in answer:
                coords = answer["coordinates"]
                self.user_lat = coords["latitude"]
                self.user_lng = coords["longitude"]
                return self.user_lat, self.user_lng
        
        # Alternative method: search for the place directly
        params = {
            "engine": "google_maps",
            "q": location_query,
            "api_key": self.serpapi_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "place_results" in results:
            place = results["place_results"]
            if "coordinates" in place:
                self.user_lat = place["coordinates"]["lat"]
                self.user_lng = place["coordinates"]["lng"]
                return self.user_lat, self.user_lng
        
        raise ValueError(f"Could not find coordinates for location: {location_query}")
    
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        Returns distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def search_places(self, query: str, max_distance_km: float) -> List[Dict]:
        """
        Search for places using Google Maps API via SerpAPI
        """
        if not self.user_lat or not self.user_lng:
            raise ValueError("User location not set. Call get_user_location first.")
        
        params = {
            "engine": "google_maps",
            "q": query,
            "ll": f"@{self.user_lat},{self.user_lng},{max_distance_km*1000}m",
            "api_key": self.serpapi_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        places = []
        if "local_results" in results:
            for place in results["local_results"]:
                if "coordinates" in place:
                    distance = self.calculate_distance(
                        self.user_lat, self.user_lng,
                        place["coordinates"]["lat"], place["coordinates"]["lng"]
                    )
                    
                    if distance <= max_distance_km:
                        places.append({
                            "name": place.get("title", "Unknown"),
                            "address": place.get("address", "Unknown"),
                            "rating": place.get("rating", 0),
                            "reviews": place.get("reviews", 0),
                            "coordinates": place["coordinates"],
                            "distance": distance,
                            "place_id": place.get("place_id", ""),
                            "types": place.get("type", [])
                        })
        
        return places
    
    def get_place_details(self, place_id: str) -> Dict:
        """
        Get detailed information about a specific place
        """
        params = {
            "engine": "google_maps_reviews",
            "place_id": place_id,
            "api_key": self.serpapi_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        details = {
            "features": [],
            "description": "",
            "hours": {},
            "phone": "",
            "website": ""
        }
        
        if "place_info" in results:
            place_info = results["place_info"]
            details["description"] = place_info.get("description", "")
            details["phone"] = place_info.get("phone", "")
            details["website"] = place_info.get("website", "")
            
            # Extract features from various fields
            if "amenities" in place_info:
                details["features"].extend(place_info["amenities"])
            
            if "service_options" in place_info:
                details["features"].extend(place_info["service_options"])
                
            if "accessibility" in place_info:
                details["features"].extend(place_info["accessibility"])
        
        return details
    
    def calculate_feature_score(self, place_features: List[str], desired_features: List[str]) -> float:
        """
        Calculate how well a place matches desired features
        """
        if not desired_features:
            return 1.0
        
        # Convert to lowercase for case-insensitive matching
        place_features_lower = [f.lower() for f in place_features]
        desired_features_lower = [f.lower() for f in desired_features]
        
        matches = 0
        for desired_feature in desired_features_lower:
            for place_feature in place_features_lower:
                if desired_feature in place_feature or place_feature in desired_feature:
                    matches += 1
                    break
        
        return matches / len(desired_features)
    
    def rank_places(self, places: List[Dict], min_rating: float, 
                   desired_features: List[str], max_results: int = 5) -> List[Dict]:
        """
        Rank places based on rating, features, and distance
        """
        scored_places = []
        
        for place in places:
            if place["rating"] < min_rating:
                continue
            
            # Get detailed information about the place
            details = {"features": []}
            if place.get("place_id"):
                try:
                    details = self.get_place_details(place["place_id"])
                    time.sleep(1)  # Rate limiting
                except:
                    pass
            
            # Calculate feature score
            feature_score = self.calculate_feature_score(details["features"], desired_features)
            
            # Calculate composite score
            # Weight: Rating (40%), Features (40%), Distance (20%)
            rating_score = place["rating"] / 5.0  # Normalize to 0-1
            distance_score = 1 / (1 + place["distance"])  # Closer = higher score
            
            composite_score = (0.4 * rating_score + 
                             0.4 * feature_score + 
                             0.2 * distance_score)
            
            place_info = place.copy()
            place_info.update({
                "features": details["features"],
                "feature_score": feature_score,
                "composite_score": composite_score,
                "details": details
            })
            
            scored_places.append(place_info)
        
        # Sort by composite score (descending)
        scored_places.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return scored_places[:max_results]
    
    def format_results(self, places: List[Dict]) -> str:
        """
        Format the results for display
        """
        if not places:
            return "No places found matching your criteria."
        
        result = "🏆 TOP RECOMMENDED PLACES:\n\n"
        
        for i, place in enumerate(places, 1):
            result += f"{i}. {place['name']}\n"
            result += f"   📍 {place['address']}\n"
            result += f"   ⭐ Rating: {place['rating']}/5 ({place['reviews']} reviews)\n"
            result += f"   📏 Distance: {place['distance']:.1f} km\n"
            result += f"   🎯 Match Score: {place['composite_score']:.2f}\n"
            
            if place['features']:
                result += f"   ✨ Features: {', '.join(place['features'][:5])}\n"
            
            if place['details'].get('phone'):
                result += f"   📞 Phone: {place['details']['phone']}\n"
            
            if place['details'].get('website'):
                result += f"   🌐 Website: {place['details']['website']}\n"
            
            result += "\n"
        
        return result

def main():
    # Initialize with your SerpAPI key
    SERPAPI_KEY = "YOUR_SERPAPI_KEY_HERE"
    
    if SERPAPI_KEY == "YOUR_SERPAPI_KEY_HERE":
        print("⚠️  Please replace 'YOUR_SERPAPI_KEY_HERE' with your actual SerpAPI key")
        print("Get your free API key at: https://serpapi.com/")
        return
    
    finder = PlaceFinder(SERPAPI_KEY)
    
    try:
        # Get user inputs
        print("🌍 PLACE FINDER")
        print("=" * 50)
        
        # Get user location
        user_location = input("Enter your current location (city, address, etc.): ")
        print(f"📍 Finding coordinates for: {user_location}")
        
        lat, lng = finder.get_user_location(user_location)
        print(f"✅ Location found: {lat:.4f}, {lng:.4f}")
        
        # Get search criteria
        search_query = input("\nWhat type of place are you looking for? (e.g., restaurants, gyms, cafes): ")
        max_distance = float(input("Maximum distance willing to travel (km): "))
        min_rating = float(input("Minimum rating (1-5): "))
        
        features_input = input("Desired features (comma-separated, e.g., 'parking, wifi, wheelchair accessible'): ")
        desired_features = [f.strip() for f in features_input.split(",") if f.strip()]
        
        print(f"\n🔍 Searching for {search_query} within {max_distance}km...")
        
        # Search for places
        places = finder.search_places(search_query, max_distance)
        print(f"📋 Found {len(places)} places")
        
        if not places:
            print("❌ No places found. Try expanding your search criteria.")
            return
        
        # Rank places
        print("🏆 Ranking places based on your preferences...")
        top_places = finder.rank_places(places, min_rating, desired_features)
        
        # Display results
        print("\n" + "=" * 50)
        print(finder.format_results(top_places))
        
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()