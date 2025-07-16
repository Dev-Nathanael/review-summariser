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

# Page configuration
st.set_page_config(
    page_title="Google Reviews Analyzer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GoogleReviewsAnalyzer:
    
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.summarizer = None
        # Initialize the summarizer with the correct method reference
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
    
    def extract_place_id_from_url(self, google_url: str) -> Optional[str]:
        """Extract place_id from Google Maps URL"""
        try:
            # Different Google Maps URL patterns
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
                        # This is a hex format, we need to convert it
                        return f"0x{extracted_id}"
                    return extracted_id
            
            # Try parsing as standard URL parameters
            parsed_url = urlparse(google_url)
            query_params = parse_qs(parsed_url.query)
            if 'place_id' in query_params:
                return query_params['place_id'][0]
            
            # If all else fails, try to use the business name from the URL for a search
            # Extract business name from URL path
            if '/place/' in google_url:
                path_parts = google_url.split('/place/')
                if len(path_parts) > 1:
                    business_name_encoded = path_parts[1].split('/')[0]
                    # Decode URL encoding
                    business_name = urllib.parse.unquote(business_name_encoded)
                    # Return a special marker indicating we should search by name
                    return f"SEARCH:{business_name}"
            
            return None
            
        except Exception as e:
            st.error(f"Error extracting place ID: {e}")
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
        """Fetch reviews for a specific place ID"""
        all_reviews = []
        next_page_token = None
        pages_fetched = 0
        max_pages = max(1, (max_reviews + 7) // 8)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while len(all_reviews) < max_reviews and pages_fetched < max_pages:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": self.serpapi_key
            }
            
            if next_page_token:
                params["next_page_token"] = next_page_token
            
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
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Successfully fetched {len(all_reviews)} reviews!")
        
        return all_reviews[:max_reviews]
    
    def summarize_reviews(self, reviews: List[Dict]) -> str:
        """Create AI summary of reviews"""
        if not reviews:
            return "No reviews to summarize"
        
        if not self.summarizer:
            return self._create_basic_summary(reviews)
        
        # Combine review texts
        review_texts = [review['text'] for review in reviews if review['text'].strip()]
        combined_text = " ".join(review_texts)
        
        if not combined_text.strip():
            return "No meaningful review content to summarize"
        
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
    
    def _create_basic_summary(self, reviews: List[Dict]) -> str:
        """Fallback summary method"""
        if not reviews:
            return "No reviews to analyze"
        
        ratings = [r['rating'] for r in reviews if r['rating'] > 0]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointing']
        
        all_text = ' '.join([r['text'].lower() for r in reviews])
        
        pos_count = sum(all_text.count(word) for word in positive_words)
        neg_count = sum(all_text.count(word) for word in negative_words)
        
        sentiment = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
        
        return f"Based on {len(reviews)} reviews, this business has an average rating of {avg_rating:.1f}/5 with generally {sentiment} customer feedback. Reviews mention various aspects of service, quality, and customer experience."

def create_map(place_data: Dict, user_location: Dict = None) -> folium.Map:
    """Create interactive map showing business and user location"""
    
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
    
    # Add business marker
    folium.Marker(
        [lat, lng],
        popup=f"<b>{place_data.get('title', 'Business')}</b><br>{place_data.get('address', '')}",
        tooltip=place_data.get('title', 'Business Location'),
        icon=folium.Icon(color='red', icon='store')
    ).add_to(m)
    
    # Add user location if available
    if user_location and user_location.get('latitude') and user_location.get('longitude'):
        user_lat = user_location['latitude']
        user_lng = user_location['longitude']
        
        folium.Marker(
            [user_lat, user_lng],
            popup="<b>Your Location</b>",
            tooltip="You are here",
            icon=folium.Icon(color='blue', icon='user')
        ).add_to(m)
        
        # Add line between user and business
        folium.PolyLine(
            locations=[[user_lat, user_lng], [lat, lng]],
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
    ratings = [r['rating'] for r in reviews if r['rating'] > 0]
    rating_counts = pd.Series(ratings).value_counts().sort_index()
    
    fig1 = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
        title='Rating Distribution',
        color=rating_counts.values,
        color_continuous_scale='RdYlGn'
    )
    fig1.update_layout(showlegend=False)
    
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
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Number of reviews
        num_reviews = st.slider("Number of reviews to fetch", 5, 100, 20)
        
        # Auto-detect user location
        st.subheader("📍 Your Location")
        
        # Location detection buttons
        col1, col2 = st.columns(2)
        with col1:
            detect_location = st.button("🌍 Detect My Location", help="Use browser geolocation")
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
            
            # Quick location presets for testing
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
            else:
                st.caption("Source: Auto-detected")
        else:
            st.info("📍 No location set (optional)")
            if not st.session_state.get('show_location_js', False):
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
                help="Be specific with location for better results"
            )
            google_url = None
        else:
            google_url = st.text_input(
                "Paste Google Maps link:",
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
                user_location = st.session_state.get('user_location')
                search_results = analyzer.search_place_with_location(query, user_location)
                
                if "error" in search_results:
                    st.error(f"❌ Search error: {search_results['error']}")
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
            reviews = st.session_state.reviews
            
            # Calculate stats
            avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
            positive = len([r for r in reviews if r['rating'] >= 4])
            negative = len([r for r in reviews if r['rating'] <= 2])
            
            # Display metrics
            st.metric("Average Rating", f"{avg_rating:.1f}⭐")
            st.metric("Total Reviews", len(reviews))
            st.metric("Positive Reviews", f"{positive} ({positive/len(reviews)*100:.1f}%)")
            st.metric("Negative Reviews", f"{negative} ({negative/len(reviews)*100:.1f}%)")
    
    # Results section
    if 'reviews' in st.session_state:
        reviews = st.session_state.reviews
        place_data = st.session_state.get('place_data')
        analyzer = st.session_state.analyzer
        
        st.header("📈 Analysis Results")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Summary", "📊 Charts", "🗺️ Map", "📝 All Reviews"])
        
        with tab1:
            st.subheader("🤖 AI-Generated Summary")
            summary = analyzer.summarize_reviews(reviews)
            st.write(summary)
            
            # Additional insights
            st.subheader("📊 Key Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_rating = sum(r['rating'] for r in reviews) / len(reviews)
                st.metric("Average Rating", f"{avg_rating:.1f}/5")
            
            with col2:
                positive_pct = len([r for r in reviews if r['rating'] >= 4]) / len(reviews) * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            
            with col3:
                total_chars = sum(len(r['text']) for r in reviews)
                avg_length = total_chars / len(reviews)
                st.metric("Avg Review Length", f"{avg_length:.0f} chars")
        
        with tab2:
            st.subheader("📊 Review Analytics")
            fig1, fig2 = create_rating_charts(reviews)
            
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        
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
        
        with tab4:
            st.subheader("📝 All Reviews")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                rating_filter = st.multiselect(
                    "Filter by rating:",
                    [1, 2, 3, 4, 5],
                    default=[1, 2, 3, 4, 5]
                )
            
            with col2:
                sort_by = st.selectbox("Sort by:", ["Rating (High to Low)", "Rating (Low to High)", "Most Recent"])
            
            # Apply filters
            filtered_reviews = [r for r in reviews if r['rating'] in rating_filter]
            
            # Sort reviews
            if sort_by == "Rating (High to Low)":
                filtered_reviews.sort(key=lambda x: x['rating'], reverse=True)
            elif sort_by == "Rating (Low to High)":
                filtered_reviews.sort(key=lambda x: x['rating'])
            # Most Recent is default order
            
            # Display reviews
            for i, review in enumerate(filtered_reviews, 1):
                # Ensure rating is an integer for star display
                rating = int(review.get('rating', 0))
                stars = '⭐' * rating if rating > 0 else '(No rating)'
                
                with st.expander(f"Review {i}: {review['author']} ({stars})"):
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

if __name__ == "__main__":
    # Clear any cached resources on restart
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()
    main()