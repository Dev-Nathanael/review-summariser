import streamlit as st
import requests
import json
import time
import re
import urllib.parse
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import os
from transformers import pipeline
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlparse, parse_qs
<<<<<<< HEAD
from collections import Counter
import math

# Page configuration
st.set_page_config(
    page_title="Cafe & Restaurant Review Analyzer",
    page_icon="☕",
=======

# Page configuration
st.set_page_config(
    page_title="Google Reviews Analyzer",
    page_icon="⭐",
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    layout="wide",
    initial_sidebar_state="expanded"
)

<<<<<<< HEAD
class CafeRestaurantAnalyzer:
=======
class GoogleReviewsAnalyzer:
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.summarizer = None
<<<<<<< HEAD
=======
        # Initialize the summarizer with the correct method reference
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        self.summarizer = self._load_summarizer()
    
    @st.cache_resource
    def _load_summarizer(_self):
        """Load the summarization model with caching"""
        try:
            with st.spinner("🤖 Loading AI summarization model..."):
                summarizer = pipeline(
                    "summarization", 
                    model="sshleifer/distilbart-cnn-12-6",  # Smaller, faster model
                    device=-1  # Use CPU
                )
            st.success("✅ AI model loaded successfully!")
            return summarizer
        except Exception as e:
            st.error(f"❌ Failed to load AI model: {e}")
            return None
    
<<<<<<< HEAD
    def resolve_google_maps_shortlink(self, short_url: str) -> Optional[str]:
        """Resolve shortened Google Maps URLs"""
        try:
            response = requests.get(short_url, allow_redirects=True, timeout=10)
            resolved_url = response.url
            st.info(f"🔗 Resolved short URL to full URL")
            return resolved_url
        except requests.RequestException as e:
            st.warning(f"❌ Failed to resolve short URL: {e}")
            return None
    
    def extract_place_id_from_url(self, google_url: str) -> Optional[str]:
        """Extract place_id from Google Maps URL with multiple fallback patterns"""
        try:
            # Handle short URLs
            if "maps.app.goo.gl" in google_url or "goo.gl" in google_url:
                resolved_url = self.resolve_google_maps_shortlink(google_url)
                if resolved_url:
                    google_url = resolved_url
                else:
                    return None
            
            # Different Google Maps URL patterns for place_id extraction
=======
    def extract_place_id_from_url(self, google_url: str) -> Optional[str]:
        """Extract place_id from Google Maps URL"""
        try:
            # Different Google Maps URL patterns
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            patterns = [
                r'place_id=([a-zA-Z0-9_-]+)',  # Standard place_id parameter
                r'/place/[^/]+/data=.*?1s0x[a-f0-9]+:0x([a-f0-9]+)',  # Hex format in data
                r'1s0x[a-f0-9]+:0x([a-f0-9]+)',  # Direct hex format
                r'data=.*?1s([a-zA-Z0-9_-]{20,})',  # Data parameter format
                r'ftid=([a-zA-Z0-9_-]+)',  # Feature ID
                r'/maps/place/[^/]+/@[^/]+/data=.*?1s0x[a-f0-9]+:0x([a-f0-9]+)',  # Full path with coordinates
            ]
            
            for pattern in patterns:
                match = re.search(pattern, google_url)
                if match:
                    extracted_id = match.group(1)
                    # Convert hex format to proper place_id if needed
                    if re.match(r'^[a-f0-9]+$', extracted_id):
<<<<<<< HEAD
=======
                        # This is a hex format, we need to convert it
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
                        return f"0x{extracted_id}"
                    return extracted_id
            
            # Try parsing as standard URL parameters
            parsed_url = urlparse(google_url)
            query_params = parse_qs(parsed_url.query)
            if 'place_id' in query_params:
                return query_params['place_id'][0]
            
<<<<<<< HEAD
            return None
            
        except Exception as e:
            st.error(f"Error extracting place ID: {e}")
            return None
    
    def extract_place_name_from_url(self, google_url: str) -> Optional[str]:
        """Extract business name from Google Maps URL as fallback"""
        try:
            # Handle short URLs
            if "maps.app.goo.gl" in google_url or "goo.gl" in google_url:
                resolved_url = self.resolve_google_maps_shortlink(google_url)
                if resolved_url:
                    google_url = resolved_url
                else:
                    return None
            
            parsed_url = urlparse(google_url)
            if "google.com/maps" in parsed_url.netloc:
                path_parts = parsed_url.path.split('/')
                if "place" in path_parts:
                    idx = path_parts.index("place")
                    if idx + 1 < len(path_parts):
                        name_part = path_parts[idx + 1]
                        # Clean up the name
                        clean_name = urllib.parse.unquote(name_part)
                        clean_name = clean_name.replace('+', ' ').replace('-', ' ')
                        return clean_name
            
            # Alternative: extract from URL path for /place/ URLs
=======
            # If all else fails, try to use the business name from the URL for a search
            # Extract business name from URL path
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            if '/place/' in google_url:
                path_parts = google_url.split('/place/')
                if len(path_parts) > 1:
                    business_name_encoded = path_parts[1].split('/')[0]
<<<<<<< HEAD
                    business_name = urllib.parse.unquote(business_name_encoded)
                    return business_name
=======
                    # Decode URL encoding
                    business_name = urllib.parse.unquote(business_name_encoded)
                    # Return a special marker indicating we should search by name
                    return f"SEARCH:{business_name}"
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            
            return None
            
        except Exception as e:
<<<<<<< HEAD
            st.error(f"Error extracting place name: {e}")
=======
            st.error(f"Error extracting place ID: {e}")
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            return None
    
    def search_place_with_location(self, query: str, user_location: Optional[Dict] = None) -> Dict:
        """Search for places using query string with optional location"""
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_maps",
            "q": query,
            "api_key": self.serpapi_key
        }
        
        # Add location-based search if user location is available
        if user_location and user_location.get('latitude') and user_location.get('longitude'):
            lat = user_location['latitude']
            lng = user_location['longitude']
<<<<<<< HEAD
            params["ll"] = f"@{lat},{lng},15z"
            st.info(f"🌍 Searching near your location ({lat:.4f}, {lng:.4f})")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with st.spinner(f"🔍 Searching for: {query}"):
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        st.warning(f"Rate limited, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                    if response.status_code != 200:
                        return {"places": [], "error": f"HTTP {response.status_code}"}
                    
                    data = response.json()
                    
                    if "error" in data:
                        return {"places": [], "error": data["error"]}
                    
                    places = []
                    if "local_results" in data:
                        for place in data["local_results"]:
                            place_info = {
                                "title": place.get("title", ""),
                                "place_id": place.get("place_id", ""),
                                "rating": place.get("rating", 0),
                                "reviews": place.get("reviews", 0),
                                "address": place.get("address", ""),
                                "type": place.get("type", ""),
                                "gps_coordinates": place.get("gps_coordinates", {}),
                                "phone": place.get("phone", ""),
                                "website": place.get("website", "")
                            }
                            
                            # Calculate distance if user location is available
                            if user_location and place_info["gps_coordinates"]:
                                place_lat = place_info["gps_coordinates"].get("latitude")
                                place_lng = place_info["gps_coordinates"].get("longitude")
                                if place_lat and place_lng:
                                    distance = self._calculate_distance(
                                        user_location['latitude'], user_location['longitude'],
                                        place_lat, place_lng
                                    )
                                    place_info["distance_km"] = distance
                            
                            places.append(place_info)
                        
                        # Sort by distance if available, otherwise by rating
                        if user_location and places and any(p.get("distance_km") for p in places):
                            places.sort(key=lambda x: x.get("distance_km", float('inf')))
                            st.success(f"✅ Found {len(places)} places sorted by distance from you")
                        else:
                            places.sort(key=lambda x: x.get("rating", 0), reverse=True)
                    
                    return {"places": places, "search_metadata": data.get("search_metadata", {})}
                    
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    return {"places": [], "error": str(e)}
                time.sleep(2 ** attempt)
                
        return {"places": [], "error": "Max retries exceeded"}
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
=======
            # Use location for more relevant results
            params["ll"] = f"@{lat},{lng},15z"  # 15z is zoom level
            st.info(f"🌍 Searching near your location ({lat:.4f}, {lng:.4f})")
        
        try:
            with st.spinner(f"🔍 Searching for: {query}"):
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    return {"places": [], "error": f"HTTP {response.status_code}"}
                
                data = response.json()
                
                if "error" in data:
                    return {"places": [], "error": data["error"]}
                
                places = []
                if "local_results" in data:
                    for place in data["local_results"]:
                        place_info = {
                            "title": place.get("title", ""),
                            "place_id": place.get("place_id", ""),
                            "rating": place.get("rating", 0),
                            "reviews": place.get("reviews", 0),
                            "address": place.get("address", ""),
                            "type": place.get("type", ""),
                            "gps_coordinates": place.get("gps_coordinates", {}),
                            "phone": place.get("phone", ""),
                            "website": place.get("website", "")
                        }
                        
                        # Calculate distance if user location is available
                        if user_location and place_info["gps_coordinates"]:
                            place_lat = place_info["gps_coordinates"].get("latitude")
                            place_lng = place_info["gps_coordinates"].get("longitude")
                            if place_lat and place_lng:
                                distance = self._calculate_distance(
                                    user_location['latitude'], user_location['longitude'],
                                    place_lat, place_lng
                                )
                                place_info["distance_km"] = distance
                        
                        places.append(place_info)
                    
                    # Sort by distance if available, otherwise by rating
                    if user_location and places and any(p.get("distance_km") for p in places):
                        places.sort(key=lambda x: x.get("distance_km", float('inf')))
                        st.success(f"✅ Found {len(places)} places sorted by distance from you")
                    else:
                        places.sort(key=lambda x: x.get("rating", 0), reverse=True)
                
                return {"places": places, "search_metadata": data.get("search_metadata", {})}
                
        except requests.RequestException as e:
            return {"places": [], "error": str(e)}
        except json.JSONDecodeError as e:
            return {"places": [], "error": "Invalid JSON response"}
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        import math
        
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        # Convert to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return round(r * c, 2)

    def get_reviews_by_place_id(self, place_id: str, max_reviews: int = 50) -> List[Dict]:
<<<<<<< HEAD
        """Fetch reviews for a specific place ID with advanced error handling"""
=======
        """Fetch reviews for a specific place ID"""
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        all_reviews = []
        next_page_token = None
        pages_fetched = 0
        max_pages = max(1, (max_reviews + 7) // 8)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
<<<<<<< HEAD
        max_retries = 3
        
=======
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        while len(all_reviews) < max_reviews and pages_fetched < max_pages:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": self.serpapi_key
            }
            
            if next_page_token:
                params["next_page_token"] = next_page_token
            
<<<<<<< HEAD
            for attempt in range(max_retries):
                try:
                    status_text.text(f"📝 Fetching reviews page {pages_fetched + 1}...")
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        status_text.text(f"⏳ Rate limited, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                    if response.status_code != 200:
                        st.error(f"❌ HTTP Error {response.status_code}")
                        break
                    
                    data = response.json()
                    
                    if "error" in data:
                        st.error(f"❌ API Error: {data['error']}")
                        break
                    
                    page_reviews = []
                    if "reviews" in data:
                        for review in data["reviews"]:
                            review_text = review.get("snippet", "") or review.get("text", "")
                            
                            page_reviews.append({
                                "place_id": place_id,
                                "author": review.get("user", {}).get("name", "Anonymous"),
                                "rating": review.get("rating", 0),
                                "text": review_text,
                                "date": review.get("date", ""),
                                "relative_date": review.get("relative_date", ""),
                                "likes": review.get("likes", 0),
                                "source": "Google Reviews (SerpAPI)"
                            })
                    
                    all_reviews.extend(page_reviews)
                    pages_fetched += 1
                    
                    # Update progress
                    progress = min(len(all_reviews) / max_reviews, 1.0)
                    progress_bar.progress(progress)
                    
                    search_metadata = data.get("search_metadata", {})
                    next_page_token = search_metadata.get("next_page_token")
                    
                    if not next_page_token or len(all_reviews) >= max_reviews:
                        break
                    
                    time.sleep(2)  # Rate limiting
                    break  # Success, exit retry loop
                    
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        st.error(f"❌ Network error: {e}")
                        break
                    time.sleep(2 ** attempt)
=======
            try:
                status_text.text(f"📝 Fetching reviews page {pages_fetched + 1}...")
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    st.error(f"❌ HTTP Error {response.status_code}")
                    break
                
                data = response.json()
                
                if "error" in data:
                    st.error(f"❌ API Error: {data['error']}")
                    break
                
                page_reviews = []
                if "reviews" in data:
                    for review in data["reviews"]:
                        review_text = review.get("snippet", "") or review.get("text", "")
                        
                        page_reviews.append({
                            "place_id": place_id,
                            "author": review.get("user", {}).get("name", "Anonymous"),
                            "rating": review.get("rating", 0),
                            "text": review_text,
                            "date": review.get("date", ""),
                            "relative_date": review.get("relative_date", ""),
                            "likes": review.get("likes", 0),
                            "source": "Google Reviews (SerpAPI)"
                        })
                
                all_reviews.extend(page_reviews)
                pages_fetched += 1
                
                # Update progress
                progress = min(len(all_reviews) / max_reviews, 1.0)
                progress_bar.progress(progress)
                
                search_metadata = data.get("search_metadata", {})
                next_page_token = search_metadata.get("next_page_token")
                
                if not next_page_token or len(all_reviews) >= max_reviews:
                    break
                
                time.sleep(1)  # Rate limiting
                
            except requests.RequestException as e:
                st.error(f"❌ Network error: {e}")
                break
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Successfully fetched {len(all_reviews)} reviews!")
        
        return all_reviews[:max_reviews]
    
<<<<<<< HEAD
    def extract_menu_items(self, combined_text: str) -> List[str]:
        """Extract popular menu items mentioned in reviews"""
        # Comprehensive cafe/restaurant menu keywords
        menu_keywords = [
            # Coffee & Drinks
            "coffee", "latte", "cappuccino", "espresso", "americano", "macchiato", 
            "mocha", "frappe", "iced coffee", "cold brew", "flat white", "cortado",
            "tea", "matcha", "chai", "smoothie", "juice", "milkshake",
            
            # Food Items
            "croissant", "bagel", "toast", "sandwich", "panini", "wrap", "burger",
            "salad", "soup", "pasta", "pizza", "cake", "muffin", "cookie", "scone",
            "avocado toast", "eggs benedict", "pancakes", "waffle", "omelette",
            "brunch", "breakfast", "lunch", "dinner",
            
            # Desserts & Snacks
            "chocolate", "vanilla", "strawberry", "cheesecake", "brownie", "donut",
            "pastry", "pie", "tart", "gelato", "ice cream", "yogurt",
            
            # Healthy Options
            "quinoa", "kale", "spinach", "organic", "gluten free", "vegan", "vegetarian"
        ]
        
        # Find all mentioned items
        found_items = []
        text_lower = combined_text.lower()
        
        for item in menu_keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(item.lower()) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                found_items.extend([item.title()] * len(matches))
        
        # Count occurrences and return top items
        item_counts = Counter(found_items)
        top_items = [item for item, count in item_counts.most_common(8)]
        
        return top_items
    
    def detect_work_friendliness(self, combined_text: str) -> Dict:
        """Detect work-friendly features from reviews"""
        text_lower = combined_text.lower()
        
        # Work-friendly indicators
        wifi_keywords = ["wifi", "wi-fi", "internet", "connection", "online"]
        power_keywords = ["power", "plug", "socket", "outlet", "charging", "charger"]
        work_keywords = ["work", "laptop", "remote", "zoom", "meeting", "study", "studying"]
        atmosphere_keywords = ["quiet", "peaceful", "calm", "focus", "concentrated"]
        negative_keywords = ["loud", "noisy", "crowded", "busy", "chaotic", "cramped"]
        
        # Count mentions
        wifi_score = sum(1 for keyword in wifi_keywords if keyword in text_lower)
        power_score = sum(1 for keyword in power_keywords if keyword in text_lower)
        work_score = sum(1 for keyword in work_keywords if keyword in text_lower)
        atmosphere_score = sum(1 for keyword in atmosphere_keywords if keyword in text_lower)
        negative_score = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        # Calculate overall work-friendliness
        total_positive = wifi_score + power_score + work_score + atmosphere_score
        work_friendly_score = max(0, total_positive - negative_score)
        
        # Determine status
        if work_friendly_score >= 3:
            status = "✅ Highly Work-Friendly"
        elif work_friendly_score >= 1:
            status = "⚠️ Moderately Work-Friendly"
        else:
            status = "❌ Limited Work-Friendly Features"
        
        return {
            "status": status,
            "wifi_mentions": wifi_score,
            "power_mentions": power_score,
            "work_mentions": work_score,
            "atmosphere_score": atmosphere_score,
            "negative_score": negative_score,
            "total_score": work_friendly_score
        }
    
    def summarize_reviews_specialized(self, reviews: List[Dict]) -> str:
        """Create specialized cafe/restaurant summary with menu items and work-friendliness"""
        if not reviews:
            return "No reviews to summarize"
        
=======
    def summarize_reviews(self, reviews: List[Dict]) -> str:
        """Create AI summary of reviews"""
        if not reviews:
            return "No reviews to summarize"
        
        if not self.summarizer:
            return self._create_basic_summary(reviews)
        
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        # Combine review texts
        review_texts = [review['text'] for review in reviews if review['text'].strip()]
        combined_text = " ".join(review_texts)
        
        if not combined_text.strip():
            return "No meaningful review content to summarize"
        
<<<<<<< HEAD
        # Extract specialized information
        menu_items = self.extract_menu_items(combined_text)
        work_info = self.detect_work_friendliness(combined_text)
        
        # Generate AI summary
        try:
            with st.spinner("🤖 Generating specialized analysis..."):
                if self.summarizer:
                    # Limit text length for model
                    max_chars = 800
                    text_to_summarize = combined_text[:max_chars]
                    
                    if len(combined_text) > max_chars:
                        text_to_summarize += "..."
                    
                    summary_result = self.summarizer(
                        text_to_summarize, 
                        max_length=150, 
                        min_length=30, 
                        do_sample=False
                    )
                    
                    ai_summary = summary_result[0]['summary_text']
                else:
                    ai_summary = self._create_basic_summary(reviews)
                    
        except Exception as e:
            st.warning(f"AI summarization failed: {e}")
            ai_summary = self._create_basic_summary(reviews)
        
        # Format menu items
        if menu_items:
            formatted_items = '\n'.join(f"• {item}" for item in menu_items[:5])
        else:
            formatted_items = "• No specific menu items frequently mentioned"
        
        # Create comprehensive summary
        summary = f"""☕ **CAFE/RESTAURANT ANALYSIS**

📋 **Overall Summary:**
{ai_summary}

🍽️ **Popular Menu Items** (mentioned in reviews):
{formatted_items}

💻 **Work-Friendly Analysis:**
{work_info['status']}

**Features Mentioned:**
• WiFi/Internet: {work_info['wifi_mentions']} mentions
• Power/Charging: {work_info['power_mentions']} mentions  
• Work/Study: {work_info['work_mentions']} mentions
• Quiet Atmosphere: {work_info['atmosphere_score']} mentions
• Noise Concerns: {work_info['negative_score']} mentions

**Work-Friendly Score:** {work_info['total_score']}/10"""
        
        return summary
=======
        try:
            with st.spinner("🤖 Generating AI summary..."):
                # Limit text length for model
                max_chars = 800
                text_to_summarize = combined_text[:max_chars]
                
                if len(combined_text) > max_chars:
                    text_to_summarize += "..."
                
                summary_result = self.summarizer(
                    text_to_summarize, 
                    max_length=150, 
                    min_length=30, 
                    do_sample=False
                )
                
                return summary_result[0]['summary_text']
                
        except Exception as e:
            st.warning(f"AI summarization failed: {e}")
            return self._create_basic_summary(reviews)
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    
    def _create_basic_summary(self, reviews: List[Dict]) -> str:
        """Fallback summary method"""
        if not reviews:
            return "No reviews to analyze"
        
        ratings = [r['rating'] for r in reviews if r['rating'] > 0]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
<<<<<<< HEAD
        positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful', 'delicious']
=======
        positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful']
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointing']
        
        all_text = ' '.join([r['text'].lower() for r in reviews])
        
        pos_count = sum(all_text.count(word) for word in positive_words)
        neg_count = sum(all_text.count(word) for word in negative_words)
        
        sentiment = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
        
<<<<<<< HEAD
        return f"Based on {len(reviews)} reviews, this establishment has an average rating of {avg_rating:.1f}/5 with generally {sentiment} customer feedback. Reviews mention various aspects of food quality, service, atmosphere, and overall experience."

def format_cafe_option(place: Dict) -> str:
    """Format cafe information for dropdown display"""
    # Get basic info
    name = place.get('title', 'Unknown')
    rating = place.get('rating', 0)
    reviews = place.get('reviews', 0)
    address = place.get('address', 'Address not available')
    
    # Format rating with stars
    if rating > 0:
        stars = '⭐' * min(int(rating), 5)  # Cap at 5 stars
        rating_text = f"{stars} {rating}/5"
    else:
        rating_text = "No rating"
    
    # Format distance if available
    distance_text = ""
    if place.get('distance_km'):
        distance_text = f" • 📍 {place['distance_km']}km away"
    
    # Create formatted string
    return f"☕ {name} • {rating_text} • ({reviews} reviews){distance_text} • {address}"

def create_map(place_data: Dict, user_location: Dict = None) -> folium.Map:
    """Create interactive map showing cafe/restaurant and user location"""
=======
        return f"Based on {len(reviews)} reviews, this business has an average rating of {avg_rating:.1f}/5 with generally {sentiment} customer feedback. Reviews mention various aspects of service, quality, and customer experience."

def create_map(place_data: Dict, user_location: Dict = None) -> folium.Map:
    """Create interactive map showing business and user location"""
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    
    # Get business coordinates
    gps = place_data.get("gps_coordinates", {})
    if not gps:
        return None
    
    lat = gps.get("latitude", 0)
    lng = gps.get("longitude", 0)
    
    if not lat or not lng:
        return None
    
    # Create map centered on business
    m = folium.Map(location=[lat, lng], zoom_start=15)
    
<<<<<<< HEAD
    # Add business marker with custom icon
    folium.Marker(
        [lat, lng],
        popup=f"<b>☕ {place_data.get('title', 'Cafe/Restaurant')}</b><br>{place_data.get('address', '')}<br>⭐ {place_data.get('rating', 'N/A')} ({place_data.get('reviews', 0)} reviews)",
        tooltip=place_data.get('title', 'Cafe/Restaurant Location'),
        icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
=======
    # Add business marker
    folium.Marker(
        [lat, lng],
        popup=f"<b>{place_data.get('title', 'Business')}</b><br>{place_data.get('address', '')}",
        tooltip=place_data.get('title', 'Business Location'),
        icon=folium.Icon(color='red', icon='store')
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    ).add_to(m)
    
    # Add user location if available
    if user_location and user_location.get('latitude') and user_location.get('longitude'):
        user_lat = user_location['latitude']
        user_lng = user_location['longitude']
        
        folium.Marker(
            [user_lat, user_lng],
<<<<<<< HEAD
            popup="<b>📍 Your Location</b>",
            tooltip="You are here",
            icon=folium.Icon(color='blue', icon='user', prefix='fa')
=======
            popup="<b>Your Location</b>",
            tooltip="You are here",
            icon=folium.Icon(color='blue', icon='user')
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        ).add_to(m)
        
        # Add line between user and business
        folium.PolyLine(
            locations=[[user_lat, user_lng], [lat, lng]],
<<<<<<< HEAD
            weight=3,
            color='blue',
            opacity=0.7,
            popup=f"Distance: {place_data.get('distance_km', 'Unknown')} km"
        ).add_to(m)
        
        # Adjust map bounds to show both locations
        m.fit_bounds([[user_lat, user_lng], [lat, lng]])
    
    return m

def create_specialized_charts(reviews: List[Dict], analyzer: CafeRestaurantAnalyzer):
    """Create specialized visualizations for cafe/restaurant data"""
    if not reviews:
        return None, None, None
    
    # 1. Rating distribution
=======
            weight=2,
            color='blue',
            opacity=0.7
        ).add_to(m)
    
    return m

def create_rating_charts(reviews: List[Dict]):
    """Create visualizations for review data"""
    if not reviews:
        return None, None
    
    # Rating distribution
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    ratings = [r['rating'] for r in reviews if r['rating'] > 0]
    rating_counts = pd.Series(ratings).value_counts().sort_index()
    
    fig1 = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
<<<<<<< HEAD
        title='⭐ Customer Rating Distribution',
=======
        title='Rating Distribution',
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        color=rating_counts.values,
        color_continuous_scale='RdYlGn'
    )
    fig1.update_layout(showlegend=False)
    
<<<<<<< HEAD
    # 2. Menu items popularity
    all_text = ' '.join([r['text'] for r in reviews])
    menu_items = analyzer.extract_menu_items(all_text)
    
    if menu_items:
        item_counts = Counter(menu_items)
        top_items = dict(item_counts.most_common(8))
        
        fig2 = px.bar(
            x=list(top_items.values()),
            y=list(top_items.keys()),
            orientation='h',
            labels={'x': 'Mentions in Reviews', 'y': 'Menu Items'},
            title='🍽️ Most Mentioned Menu Items',
            color=list(top_items.values()),
            color_continuous_scale='Viridis'
        )
        fig2.update_layout(showlegend=False, height=400)
    else:
        fig2 = None
    
    # 3. Work-friendliness analysis
    work_info = analyzer.detect_work_friendliness(all_text)
    
    work_features = {
        'WiFi/Internet': work_info['wifi_mentions'],
        'Power/Charging': work_info['power_mentions'],
        'Work/Study': work_info['work_mentions'],
        'Quiet Atmosphere': work_info['atmosphere_score']
    }
    
    if any(work_features.values()):
        fig3 = px.bar(
            x=list(work_features.keys()),
            y=list(work_features.values()),
            labels={'x': 'Work-Friendly Features', 'y': 'Mentions in Reviews'},
            title='💻 Work-Friendly Features Analysis',
            color=list(work_features.values()),
            color_continuous_scale='Blues'
        )
        fig3.update_layout(showlegend=False)
    else:
        fig3 = None
    
    return fig1, fig2, fig3

def main():
    st.title("☕ Cafe & Restaurant Review Analyzer")
    st.markdown("Specialized analysis for cafes and restaurants with menu insights and work-friendliness detection")
=======
    # Review timeline (if dates available)
    dates = [r['date'] for r in reviews if r['date']]
    if dates:
        try:
            # Simple month/year grouping
            date_series = pd.to_datetime(dates, errors='coerce').dropna()
            if len(date_series) > 0:
                monthly_counts = date_series.dt.to_period('M').value_counts().sort_index()
                
                fig2 = px.line(
                    x=monthly_counts.index.astype(str),
                    y=monthly_counts.values,
                    labels={'x': 'Month', 'y': 'Number of Reviews'},
                    title='Reviews Over Time'
                )
                fig2.update_layout(xaxis_tickangle=-45)
                return fig1, fig2
        except:
            pass
    
    return fig1, None

def main():
    st.title("⭐ Google Reviews Analyzer")
    st.markdown("Analyze customer reviews for any business using AI-powered insights")
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    
    # Check for API key in environment only (hidden from users)
    api_key = os.getenv("SERPAPI", "")
    if not api_key:
        st.error("🔑 **Configuration Error**: SerpAPI key not found in environment variables.")
        st.markdown("""
        **For Administrators**: Please set the SERPAPI environment variable:
        ```bash
        export SERPAPI=your_serpapi_key_here
        ```
        """)
        st.stop()
    
<<<<<<< HEAD
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_place' not in st.session_state:
        st.session_state.selected_place = None
    if 'reviews' not in st.session_state:
        st.session_state.reviews = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
=======
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Number of reviews
<<<<<<< HEAD
        num_reviews = st.slider("Number of reviews to fetch", 5, 100, 25)
=======
        num_reviews = st.slider("Number of reviews to fetch", 5, 100, 20)
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        
        # Auto-detect user location
        st.subheader("📍 Your Location")
        
        # Location detection buttons
        col1, col2 = st.columns(2)
        with col1:
<<<<<<< HEAD
            detect_location = st.button("🌍 Detect Location", help="Use browser geolocation")
=======
            detect_location = st.button("🌍 Detect My Location", help="Use browser geolocation")
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        with col2:
            clear_location = st.button("🗑️ Clear Location")
        
        # Initialize session state for location
        if 'user_location' not in st.session_state:
            st.session_state.user_location = None
        
        # Handle location detection
        if detect_location:
            st.session_state.show_location_js = True
        
        if clear_location:
            st.session_state.user_location = None
            st.session_state.show_location_js = False
            st.rerun()
        
        # Show location detection interface
        if st.session_state.get('show_location_js', False):
<<<<<<< HEAD
            st.markdown("🔍 **Detecting your location...**")
            st.caption("Please allow location access when prompted.")
            
            # HTML/JS component for geolocation
            st.components.v1.html(f"""
            <div id="location-container">
                <div id="location-status">Requesting location...</div>
            </div>
            
            <script>
            function detectLocation() {{
                const statusDiv = document.getElementById('location-status');
                
                if (navigator.geolocation) {{
                    statusDiv.innerHTML = 'Getting your location...';
                    
                    navigator.geolocation.getCurrentPosition(
                        function(position) {{
                            const lat = position.coords.latitude;
                            const lng = position.coords.longitude;
                            
                            statusDiv.innerHTML = `
                                <div style="color: green;">
                                    ✅ <strong>Location detected!</strong><br>
                                    📍 ${{lat.toFixed(6)}}, ${{lng.toFixed(6)}}<br>
                                    <small>Will show distance to cafes/restaurants.</small>
                                </div>
                            `;
                        }},
                        function(error) {{
                            let errorMsg = '';
                            switch(error.code) {{
                                case error.PERMISSION_DENIED:
                                    errorMsg = "❌ Location access denied.";
                                    break;
                                case error.POSITION_UNAVAILABLE:
                                    errorMsg = "❌ Location unavailable.";
                                    break;
                                case error.TIMEOUT:
                                    errorMsg = "❌ Location request timed out.";
                                    break;
                                default:
                                    errorMsg = "❌ Unknown error occurred.";
                                    break;
                            }}
                            statusDiv.innerHTML = `<div style="color: orange;">${{errorMsg}}</div>`;
                        }}
                    );
                }} else {{
                    statusDiv.innerHTML = '<div style="color: red;">❌ Geolocation not supported.</div>';
                }}
            }}
            
            detectLocation();
            </script>
            """, height=120)
            
            # Manual location input as fallback
            st.markdown("---")
            st.markdown("**📍 Manual Entry**")
=======
            # Use Streamlit components for better integration
            location_placeholder = st.empty()
            
            with location_placeholder.container():
                st.markdown("🔍 **Detecting your location...**")
                st.caption("Please allow location access when prompted by your browser.")
                
                # Create a unique key for this session
                location_key = f"location_{int(time.time())}"
                
                # HTML/JS component for geolocation
                st.components.v1.html(f"""
                <div id="location-container">
                    <div id="location-status">Requesting location...</div>
                </div>
                
                <script>
                function detectLocation() {{
                    const statusDiv = document.getElementById('location-status');
                    
                    if (navigator.geolocation) {{
                        statusDiv.innerHTML = 'Getting your location...';
                        
                        navigator.geolocation.getCurrentPosition(
                            function(position) {{
                                const lat = position.coords.latitude;
                                const lng = position.coords.longitude;
                                
                                statusDiv.innerHTML = `
                                    <div style="color: green;">
                                        ✅ <strong>Location detected!</strong><br>
                                        📍 ${{lat.toFixed(6)}}, ${{lng.toFixed(6)}}<br>
                                        <small>Location will be used for distance calculations.</small>
                                    </div>
                                `;
                                
                                // Send data back to Streamlit
                                window.parent.postMessage({{
                                    type: 'location_detected',
                                    latitude: lat,
                                    longitude: lng
                                }}, '*');
                            }},
                            function(error) {{
                                let errorMsg = '';
                                switch(error.code) {{
                                    case error.PERMISSION_DENIED:
                                        errorMsg = "❌ Location access denied. You can still use the app without location features.";
                                        break;
                                    case error.POSITION_UNAVAILABLE:
                                        errorMsg = "❌ Location information unavailable.";
                                        break;
                                    case error.TIMEOUT:
                                        errorMsg = "❌ Location request timed out.";
                                        break;
                                    default:
                                        errorMsg = "❌ An unknown error occurred.";
                                        break;
                                }}
                                statusDiv.innerHTML = `<div style="color: orange;">${{errorMsg}}</div>`;
                            }}
                        );
                    }} else {{
                        statusDiv.innerHTML = '<div style="color: red;">❌ Geolocation is not supported by this browser.</div>';
                    }}
                }}
                
                // Auto-detect on load
                detectLocation();
                </script>
                """, height=120)
            
            # Check if location was provided via query params (simple fallback)
            try:
                query_params = st.query_params
                if 'lat' in query_params and 'lng' in query_params:
                    lat = float(query_params['lat'])
                    lng = float(query_params['lng'])
                    st.session_state.user_location = {
                        "latitude": lat,
                        "longitude": lng,
                        "source": "browser"
                    }
                    st.session_state.show_location_js = False
                    st.success("📍 Location detected successfully!")
                    st.rerun()
            except:
                pass
            
            # Manual location input as fallback
            st.markdown("---")
            st.markdown("**🔧 Manual Entry (Alternative)**")
            st.caption("If automatic detection doesn't work, you can enter coordinates manually:")
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            
            col1, col2 = st.columns(2)
            with col1:
                manual_lat = st.number_input("Latitude", value=0.0, format="%.6f", key="manual_lat")
            with col2:
                manual_lng = st.number_input("Longitude", value=0.0, format="%.6f", key="manual_lng")
            
            if st.button("📍 Use Manual Location"):
                if manual_lat != 0.0 and manual_lng != 0.0:
                    st.session_state.user_location = {
                        "latitude": manual_lat, 
                        "longitude": manual_lng,
                        "source": "manual"
                    }
                    st.session_state.show_location_js = False
                    st.success("📍 Manual location set!")
                    st.rerun()
                else:
                    st.warning("Please enter valid coordinates")
            
<<<<<<< HEAD
            # Quick location presets
=======
            # Quick location presets for testing
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            st.markdown("**🌍 Quick Locations**")
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            
            with preset_col1:
                if st.button("🇺🇸 NYC", help="New York City"):
                    st.session_state.user_location = {
                        "latitude": 40.7128, "longitude": -74.0060, "source": "preset"
                    }
                    st.session_state.show_location_js = False
                    st.rerun()
            
            with preset_col2:
                if st.button("🇮🇩 Jakarta", help="Jakarta, Indonesia"):
                    st.session_state.user_location = {
                        "latitude": -6.2088, "longitude": 106.8456, "source": "preset"
                    }
                    st.session_state.show_location_js = False
                    st.rerun()
            
            with preset_col3:
                if st.button("🇬🇧 London", help="London, UK"):
                    st.session_state.user_location = {
                        "latitude": 51.5074, "longitude": -0.1278, "source": "preset"
                    }
                    st.session_state.show_location_js = False
                    st.rerun()
        
        # Display current location status
        if st.session_state.user_location:
            st.success("📍 Location Available")
            loc = st.session_state.user_location
            st.info(f"📍 {loc['latitude']:.4f}, {loc['longitude']:.4f}")
            if loc.get('source') == 'manual':
                st.caption("Source: Manual entry")
<<<<<<< HEAD
            elif loc.get('source') == 'preset':
                st.caption("Source: Quick location")
=======
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            else:
                st.caption("Source: Auto-detected")
        else:
            st.info("📍 No location set (optional)")
            if not st.session_state.get('show_location_js', False):
<<<<<<< HEAD
                st.caption("Click 'Detect Location' to enable distance features")
    
    # Main content area
    st.header("🔍 Find Cafe/Restaurant")
    
    # Step 1: Search for cafes
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input method selection
        input_method = st.radio(
            "Choose search method:",
            ["Cafe/Restaurant Name", "Google Maps Link"],
            horizontal=True
        )
        
        if input_method == "Cafe/Restaurant Name":
            query = st.text_input(
                "Enter cafe or restaurant name:",
                placeholder="e.g., Blue Bottle Coffee, Joe's Pizza, Starbucks",
=======
                st.caption("Click 'Detect My Location' to enable map features")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Business Search")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Business Name", "Google Maps Link"],
            horizontal=True
        )
        
        if input_method == "Business Name":
            query = st.text_input(
                "Enter business name:",
                placeholder="e.g., Starbucks Downtown Seattle",
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
                help="Be specific with location for better results"
            )
            google_url = None
        else:
            google_url = st.text_input(
                "Paste Google Maps link:",
<<<<<<< HEAD
                placeholder="https://maps.google.com/maps/place/...",
                help="Copy the URL from Google Maps (supports short links too)"
            )
            query = None
    
    with col2:
        # Search button
        if st.button("🔍 Search Cafes", type="primary"):
            if not st.session_state.analyzer:
                st.session_state.analyzer = CafeRestaurantAnalyzer(api_key)
            
            analyzer = st.session_state.analyzer
            
            # Handle different input methods
            if input_method == "Google Maps Link" and google_url:
                st.info("🔍 Processing Google Maps link...")
                
                # Step 1: Try to extract place_id first (preferred method)
                place_id = analyzer.extract_place_id_from_url(google_url)
                
                if place_id:
                    st.success(f"✅ Extracted place ID: {place_id}")
                    
                    # Test if place_id works by fetching 1 review
                    test_reviews = analyzer.get_reviews_by_place_id(place_id, 1)
                    if test_reviews:
                        st.success("✅ Place ID verified - reviews found!")
                        # Create a single place result
                        st.session_state.search_results = {
                            "places": [{"place_id": place_id, "title": "Selected Location", "rating": 0, "reviews": 0, "address": "From Google Maps URL"}],
                            "direct_place_id": place_id
                        }
                    else:
                        st.warning("⚠️ Place ID found but no reviews. Trying fallback method...")
                        place_id = None
                
                # Step 2: Fallback to business name extraction
                if not place_id:
                    business_name = analyzer.extract_place_name_from_url(google_url)
                    
                    if business_name:
                        st.info(f"🔍 Extracted business name: {business_name}")
                        st.info("Searching by business name...")
                        
                        user_location = st.session_state.get('user_location')
                        search_results = analyzer.search_place_with_location(business_name, user_location)
                        
                        if "error" in search_results:
                            st.error(f"❌ Search error: {search_results['error']}")
                        elif not search_results["places"]:
                            st.error(f"❌ No cafes/restaurants found for: {business_name}")
                        else:
                            st.session_state.search_results = search_results
                            st.success(f"✅ Found {len(search_results['places'])} cafes/restaurants")
                    else:
                        st.error("❌ Could not extract business information from URL. Please check the link or try the business name method.")
            
            elif input_method == "Cafe/Restaurant Name" and query:
                # Direct business name search
=======
                placeholder="https://maps.google.com/...",
                help="Copy the URL from Google Maps"
            )
            query = None
        
        # Search button
        if st.button("🔍 Analyze Reviews", type="primary"):
            if not api_key:
                st.error("Please provide your SerpAPI key in the sidebar")
                st.stop()
            
            analyzer = GoogleReviewsAnalyzer(api_key)
            
            # Initialize variables
            place_data = None
            place_id = None
            
            # Handle different input methods
            if input_method == "Google Maps Link" and google_url:
                extracted = analyzer.extract_place_id_from_url(google_url)
                if not extracted:
                    st.error("❌ Could not extract place ID from URL. Please check the link.")
                    st.stop()
                elif extracted.startswith("SEARCH:"):
                    # Extract business name and do a search instead
                    business_name = extracted.replace("SEARCH:", "")
                    st.info(f"🔍 Extracted business name: {business_name}")
                    st.info("Searching for this business...")
                    
                    user_location = st.session_state.get('user_location')
                    search_results = analyzer.search_place_with_location(business_name, user_location)
                    
                    if "error" in search_results:
                        st.error(f"❌ Search error: {search_results['error']}")
                        st.stop()
                    
                    if not search_results["places"]:
                        st.error(f"❌ No businesses found for: {business_name}")
                        st.stop()
                    
                    # Let user choose if multiple results
                    places = search_results["places"]
                    if len(places) > 1:
                        st.subheader("Multiple businesses found:")
                        
                        def format_place(i):
                            place = places[i]
                            base_info = f"{place['title']} - {place['address']} ({place['rating']}⭐"
                            if place.get('distance_km'):
                                return f"{base_info}, {place['distance_km']}km away)"
                            return f"{base_info})"
                        
                        choice = st.selectbox(
                            "Select the correct business:",
                            range(len(places)),
                            format_func=format_place
                        )
                        place_data = places[choice]
                    else:
                        place_data = places[0]
                    
                    place_id = place_data["place_id"]
                else:
                    place_id = extracted
                    st.success(f"✅ Extracted place ID: {place_id}")
                    
                    # Try to fetch reviews with the extracted place_id first
                    test_reviews = analyzer.get_reviews_by_place_id(place_id, 1)  # Test with just 1 review
                    if not test_reviews:
                        st.warning("⚠️ No reviews found with extracted place ID. Trying business name search...")
                        # Extract business name from URL and search instead
                        if '/place/' in google_url:
                            path_parts = google_url.split('/place/')
                            if len(path_parts) > 1:
                                business_name_encoded = path_parts[1].split('/')[0]
                                business_name = urllib.parse.unquote(business_name_encoded)
                                st.info(f"🔍 Searching for: {business_name}")
                                
                                user_location = st.session_state.get('user_location')
                                search_results = analyzer.search_place_with_location(business_name, user_location)
                                
                                if "error" not in search_results and search_results["places"]:
                                    places = search_results["places"]
                                    if len(places) > 1:
                                        st.subheader("Multiple businesses found:")
                                        
                                        def format_place_fallback(i):
                                            place = places[i]
                                            base_info = f"{place['title']} - {place['address']} ({place['rating']}⭐"
                                            if place.get('distance_km'):
                                                return f"{base_info}, {place['distance_km']}km away)"
                                            return f"{base_info})"
                                        
                                        choice = st.selectbox(
                                            "Select the correct business:",
                                            range(len(places)),
                                            format_func=format_place_fallback
                                        )
                                        place_data = places[choice]
                                    else:
                                        place_data = places[0]
                                    
                                    place_id = place_data["place_id"]
                                    st.success(f"🎯 Found business: {place_data['title']}")
                    else:
                        # Reset progress indicators since we did a test fetch
                        st.empty()
            
            elif input_method == "Business Name" and query:
                # Search for the business
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
                user_location = st.session_state.get('user_location')
                search_results = analyzer.search_place_with_location(query, user_location)
                
                if "error" in search_results:
                    st.error(f"❌ Search error: {search_results['error']}")
<<<<<<< HEAD
                elif not search_results["places"]:
                    st.error(f"❌ No cafes/restaurants found for: {query}")
                else:
                    st.session_state.search_results = search_results
                    st.success(f"✅ Found {len(search_results['places'])} cafes/restaurants")
            
            else:
                st.error("Please provide either a cafe/restaurant name or Google Maps link")
    
    # Step 2: Show dropdown selection if search results exist
    if st.session_state.search_results:
        st.markdown("---")
        st.header("☕ Select Your Cafe/Restaurant")
        
        places = st.session_state.search_results["places"]
        
        if len(places) == 1:
            # If only one result, auto-select it
            st.session_state.selected_place = places[0]
            selected_place = places[0]
            st.success(f"✅ Auto-selected: {selected_place.get('title', 'Selected Location')}")
        else:
            # Multiple results - show dropdown
            st.subheader("🏪 Choose from the following options:")
            
            # Create dropdown options
            dropdown_options = []
            for i, place in enumerate(places):
                option_text = format_cafe_option(place)
                dropdown_options.append(option_text)
            
            # Dropdown selection
            selected_index = st.selectbox(
                "Select your cafe/restaurant:",
                range(len(dropdown_options)),
                format_func=lambda x: dropdown_options[x],
                key="cafe_selector"
            )
            
            # Store selected place
            st.session_state.selected_place = places[selected_index]
            selected_place = places[selected_index]
            
            # Show selection confirmation
            with st.expander("📋 Selected Cafe Details", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**📍 Name:** {selected_place.get('title', 'N/A')}")
                    st.markdown(f"**⭐ Rating:** {selected_place.get('rating', 'N/A')}/5")
                    st.markdown(f"**📝 Reviews:** {selected_place.get('reviews', 'N/A')}")
                    st.markdown(f"**🏷️ Type:** {selected_place.get('type', 'N/A')}")
                
                with col2:
                    st.markdown(f"**📍 Address:** {selected_place.get('address', 'N/A')}")
                    if selected_place.get('distance_km'):
                        st.markdown(f"**📏 Distance:** {selected_place['distance_km']} km")
                    st.markdown(f"**📞 Phone:** {selected_place.get('phone', 'N/A')}")
                    if selected_place.get('website'):
                        st.markdown(f"**🌐 Website:** {selected_place['website']}")
        
        # Step 3: Analyze button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("☕ Analyze This Cafe/Restaurant", type="primary", use_container_width=True):
                selected_place = st.session_state.selected_place
                
                if not selected_place:
                    st.error("❌ No cafe selected")
                    st.stop()
                
                # Get place_id
                place_id = selected_place.get("place_id")
                
                # Handle direct place_id from Google Maps URL
                if not place_id and st.session_state.search_results.get("direct_place_id"):
                    place_id = st.session_state.search_results["direct_place_id"]
                
                if not place_id:
                    st.error("❌ No place ID found for selected cafe")
                    st.stop()
                
                # Show selected business info
                st.success(f"☕ Analyzing: {selected_place.get('title', 'Selected Location')}")
                
                # Build info string with available data
                info_parts = []
                if selected_place.get('address'):
                    info_parts.append(f"📍 {selected_place['address']}")
                if selected_place.get('rating'):
                    info_parts.append(f"⭐ {selected_place['rating']}")
                if selected_place.get('reviews'):
                    info_parts.append(f"📝 {selected_place['reviews']} reviews")
                if selected_place.get('distance_km'):
                    info_parts.append(f"📏 {selected_place['distance_km']}km away")
                
                if info_parts:
                    st.info(" | ".join(info_parts))
                
                # Fetch reviews
                st.info(f"📝 Fetching {num_reviews} reviews for analysis...")
                analyzer = st.session_state.analyzer
                reviews = analyzer.get_reviews_by_place_id(place_id, num_reviews)
                
                if not reviews:
                    st.error("❌ No reviews found for this cafe/restaurant")
                    st.info("💡 This might be because:")
                    st.info("• The business is too new")
                    st.info("• Reviews are not publicly available")
                    st.info("• Temporary API issues")
                    st.stop()
                
                # Store results in session state for persistence
                st.session_state.reviews = reviews
                st.session_state.place_data = selected_place
                
                st.success(f"✅ Successfully fetched {len(reviews)} reviews!")
                st.rerun()
    
    # Quick Stats sidebar
    if st.session_state.reviews:
        with st.sidebar:
            st.markdown("---")
            st.header("📊 Quick Stats")
=======
                    st.stop()
                
                if not search_results["places"]:
                    st.error(f"❌ No businesses found for: {query}")
                    st.stop()
                
                # Let user choose if multiple results
                places = search_results["places"]
                if len(places) > 1:
                    st.subheader("Multiple businesses found:")
                    
                    def format_business_choice(i):
                        place = places[i]
                        base_info = f"{place['title']} - {place['address']} ({place['rating']}⭐"
                        if place.get('distance_km'):
                            return f"{base_info}, {place['distance_km']}km away)"
                        return f"{base_info})"
                    
                    choice = st.selectbox(
                        "Select the correct business:",
                        range(len(places)),
                        format_func=format_business_choice
                    )
                    place_data = places[choice]
                else:
                    place_data = places[0]
                
                place_id = place_data["place_id"]
                
                if not place_id:
                    st.error("❌ No place ID found for selected business")
                    st.stop()
            
            else:
                st.error("Please provide either a business name or Google Maps link")
                st.stop()
            
            # Show selected business info
            if place_data:
                st.success(f"🎯 Selected: {place_data['title']}")
                
                # Build info string with distance if available
                info_parts = [
                    f"📍 {place_data['address']}",
                    f"⭐ {place_data['rating']}",
                    f"📝 {place_data['reviews']} reviews"
                ]
                
                if place_data.get('distance_km'):
                    info_parts.append(f"📏 {place_data['distance_km']}km away")
                
                st.info(" | ".join(info_parts))
            
            # Fetch reviews
            reviews = analyzer.get_reviews_by_place_id(place_id, num_reviews)
            
            if not reviews:
                st.error("❌ No reviews found for this business")
                st.stop()
            
            # Store results in session state for persistence
            st.session_state.reviews = reviews
            st.session_state.place_data = place_data
            st.session_state.analyzer = analyzer
    
    with col2:
        st.header("📊 Quick Stats")
        if 'reviews' in st.session_state:
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            reviews = st.session_state.reviews
            
            # Calculate stats
            avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
            positive = len([r for r in reviews if r['rating'] >= 4])
            negative = len([r for r in reviews if r['rating'] <= 2])
            
<<<<<<< HEAD
            # Work-friendliness quick check
            all_text = ' '.join([r['text'] for r in reviews])
            work_info = st.session_state.analyzer.detect_work_friendliness(all_text)
            
=======
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            # Display metrics
            st.metric("Average Rating", f"{avg_rating:.1f}⭐")
            st.metric("Total Reviews", len(reviews))
            st.metric("Positive Reviews", f"{positive} ({positive/len(reviews)*100:.1f}%)")
<<<<<<< HEAD
            st.metric("Work-Friendly Score", f"{work_info['total_score']}/10")
    
    # Results section
    if st.session_state.reviews:
=======
            st.metric("Negative Reviews", f"{negative} ({negative/len(reviews)*100:.1f}%)")
    
    # Results section
    if 'reviews' in st.session_state:
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        reviews = st.session_state.reviews
        place_data = st.session_state.get('place_data')
        analyzer = st.session_state.analyzer
        
<<<<<<< HEAD
        st.markdown("---")
        st.header("📈 Detailed Analysis")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "☕ Specialized Analysis", 
            "📊 Charts & Insights", 
            "🗺️ Location & Map", 
            "📝 All Reviews",
            "📥 Export Data"
        ])
        
        with tab1:
            st.subheader("☕ Cafe/Restaurant Specialized Analysis")
            
            # Generate specialized summary
            specialized_summary = analyzer.summarize_reviews_specialized(reviews)
            st.markdown(specialized_summary)
            
            # Additional quick insights
            st.markdown("---")
            st.subheader("🔍 Quick Insights")
            
=======
        st.header("📈 Analysis Results")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Summary", "📊 Charts", "🗺️ Map", "📝 All Reviews"])
        
        with tab1:
            st.subheader("🤖 AI-Generated Summary")
            summary = analyzer.summarize_reviews(reviews)
            st.write(summary)
            
            # Additional insights
            st.subheader("📊 Key Insights")
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
                st.metric("Average Rating", f"{avg_rating:.1f}/5")
            
            with col2:
                positive_pct = len([r for r in reviews if r['rating'] >= 4]) / len(reviews) * 100
<<<<<<< HEAD
                st.metric("Positive Reviews", f"{positive_pct:.1f}%")
            
            with col3:
                total_chars = sum(len(r['text']) for r in reviews)
                avg_length = total_chars / len(reviews) if reviews else 0
                st.metric("Avg Review Length", f"{avg_length:.0f} chars")
        
        with tab2:
            st.subheader("📊 Visual Analytics")
            
            # Create specialized charts
            fig1, fig2, fig3 = create_specialized_charts(reviews, analyzer)
=======
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            
            with col3:
                total_chars = sum(len(r['text']) for r in reviews)
                avg_length = total_chars / len(reviews)
                st.metric("Avg Review Length", f"{avg_length:.0f} chars")
        
        with tab2:
            st.subheader("📊 Review Analytics")
            fig1, fig2 = create_rating_charts(reviews)
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
<<<<<<< HEAD
            else:
                st.info("📊 No specific menu items were frequently mentioned in reviews")
            
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("💻 No work-friendly features mentioned in reviews")
            
            # Review timeline if available
            dates = [r['date'] for r in reviews if r['date']]
            if dates:
                try:
                    date_series = pd.to_datetime(dates, errors='coerce').dropna()
                    if len(date_series) > 1:
                        monthly_counts = date_series.dt.to_period('M').value_counts().sort_index()
                        
                        fig_timeline = px.line(
                            x=monthly_counts.index.astype(str),
                            y=monthly_counts.values,
                            labels={'x': 'Month', 'y': 'Number of Reviews'},
                            title='📅 Review Activity Over Time'
                        )
                        fig_timeline.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                except:
                    pass
        
        with tab3:
            st.subheader("🗺️ Location & Map")
            
            if place_data:
                # Show detailed location info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **📍 Address:** {place_data.get('address', 'N/A')}  
                    **📞 Phone:** {place_data.get('phone', 'N/A')}  
                    **🌐 Website:** {place_data.get('website', 'N/A')}  
                    **🏷️ Type:** {place_data.get('type', 'N/A')}
                    """)
                
                with col2:
                    if place_data.get('distance_km'):
                        st.metric("Distance from You", f"{place_data['distance_km']} km")
                    
                    gps = place_data.get('gps_coordinates', {})
                    if gps:
                        st.markdown(f"""
                        **🧭 Coordinates:**  
                        Lat: {gps.get('latitude', 'N/A')}  
                        Lng: {gps.get('longitude', 'N/A')}
                        """)
                
                # Interactive map
                user_location = st.session_state.get('user_location')
                map_obj = create_map(place_data, user_location)
                if map_obj:
                    st.markdown("**🗺️ Interactive Map:**")
                    folium_static(map_obj, width=700, height=500)
                else:
                    st.warning("📍 Location coordinates not available for mapping")
            else:
                st.info("📍 Location information not available")
=======
        
        with tab3:
            st.subheader("🗺️ Location Map")
            if place_data:
                user_location = st.session_state.get('user_location')
                map_obj = create_map(place_data, user_location)
                if map_obj:
                    folium_static(map_obj, width=700, height=500)
                else:
                    st.warning("📍 Location data not available for mapping")
            else:
                st.info("📍 Map not available - business was searched by place ID only")
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
        
        with tab4:
            st.subheader("📝 All Reviews")
            
<<<<<<< HEAD
            # Filter and sort options
=======
            # Filter options
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            col1, col2 = st.columns(2)
            with col1:
                rating_filter = st.multiselect(
                    "Filter by rating:",
                    [1, 2, 3, 4, 5],
                    default=[1, 2, 3, 4, 5]
                )
            
            with col2:
<<<<<<< HEAD
                sort_by = st.selectbox(
                    "Sort by:", 
                    ["Rating (High to Low)", "Rating (Low to High)", "Most Recent", "Most Liked"]
                )
=======
                sort_by = st.selectbox("Sort by:", ["Rating (High to Low)", "Rating (Low to High)", "Most Recent"])
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            
            # Apply filters
            filtered_reviews = [r for r in reviews if r['rating'] in rating_filter]
            
            # Sort reviews
            if sort_by == "Rating (High to Low)":
                filtered_reviews.sort(key=lambda x: x['rating'], reverse=True)
            elif sort_by == "Rating (Low to High)":
                filtered_reviews.sort(key=lambda x: x['rating'])
<<<<<<< HEAD
            elif sort_by == "Most Liked":
                filtered_reviews.sort(key=lambda x: x.get('likes', 0), reverse=True)
            # Most Recent is default order
            
            st.info(f"Showing {len(filtered_reviews)} of {len(reviews)} reviews")
            
=======
            # Most Recent is default order
            
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
            # Display reviews
            for i, review in enumerate(filtered_reviews, 1):
                # Ensure rating is an integer for star display
                rating = int(review.get('rating', 0))
                stars = '⭐' * rating if rating > 0 else '(No rating)'
                
                with st.expander(f"Review {i}: {review['author']} ({stars})"):
<<<<<<< HEAD
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Date:** {review['date']}")
                        st.write(f"**Rating:** {stars} ({rating}/5)")
                        st.write(f"**Review:** {review['text']}")
                    
                    with col2:
                        if review.get('likes', 0) > 0:
                            st.metric("👍 Likes", review['likes'])
        
        with tab5:
            st.subheader("📥 Export & Download")
            
            # Export options
            st.markdown("**📋 Available Export Formats:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                if st.button("📊 Download Reviews as CSV"):
                    df = pd.DataFrame(reviews)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV File",
                        data=csv,
                        file_name=f"cafe_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Analysis Summary Export
                if st.button("📄 Download Analysis Summary"):
                    summary = analyzer.summarize_reviews_specialized(reviews)
                    
                    # Create detailed report
                    report = f"""CAFE/RESTAURANT ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BUSINESS INFORMATION:
- Name: {place_data.get('title', 'N/A') if place_data else 'N/A'}
- Address: {place_data.get('address', 'N/A') if place_data else 'N/A'}
- Rating: {place_data.get('rating', 'N/A') if place_data else 'N/A'}
- Total Reviews: {place_data.get('reviews', 'N/A') if place_data else 'N/A'}

ANALYSIS SUMMARY:
{summary}

REVIEW STATISTICS:
- Reviews Analyzed: {len(reviews)}
- Average Rating: {sum(r['rating'] for r in reviews) / len(reviews):.1f}/5
- Positive (4-5 stars): {len([r for r in reviews if r['rating'] >= 4])}/{len(reviews)}
- Negative (1-2 stars): {len([r for r in reviews if r['rating'] <= 2])}/{len(reviews)}

Generated by Cafe & Restaurant Review Analyzer
"""
                    
                    st.download_button(
                        label="💾 Download Analysis Report",
                        data=report,
                        file_name=f"cafe_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                # JSON Export for developers
                if st.button("🔧 Download Raw Data (JSON)"):
                    export_data = {
                        "business_info": place_data,
                        "reviews": reviews,
                        "analysis_metadata": {
                            "generated_at": datetime.now().isoformat(),
                            "total_reviews": len(reviews),
                            "average_rating": sum(r['rating'] for r in reviews) / len(reviews) if reviews else 0
                        }
                    }
                    
                    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Download JSON File",
                        data=json_str,
                        file_name=f"cafe_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                # Share-friendly summary
                if st.button("📱 Generate Share Summary"):
                    share_text = f"""☕ {place_data.get('title', 'Cafe/Restaurant') if place_data else 'Cafe Analysis'}

⭐ Rating: {place_data.get('rating', 'N/A') if place_data else 'N/A'}/5 ({place_data.get('reviews', 'N/A') if place_data else 'N/A'} reviews)
📍 {place_data.get('address', 'Location N/A') if place_data else 'Location N/A'}

📊 Analysis of {len(reviews)} recent reviews:
• Average: {sum(r['rating'] for r in reviews) / len(reviews):.1f}/5
• Positive: {len([r for r in reviews if r['rating'] >= 4])}/{len(reviews)} reviews

Generated by Cafe & Restaurant Review Analyzer"""
                    
                    st.text_area("📋 Copy this summary to share:", share_text, height=200)
=======
                    st.write(f"**Date:** {review['date']}")
                    st.write(f"**Rating:** {stars} ({rating}/5)")
                    st.write(f"**Review:** {review['text']}")
                    if review['likes'] > 0:
                        st.write(f"**Likes:** 👍 {review['likes']}")
            
            # Download option
            if st.button("📥 Download Reviews as CSV"):
                df = pd.DataFrame(filtered_reviews)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0

if __name__ == "__main__":
    # Clear any cached resources on restart
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
    main()