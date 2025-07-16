<<<<<<< HEAD
import requests
import json
import time
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import os
from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleReviewsAnalyzer:
    
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        logger.info("Initialized with SerpAPI")
        
        # Initialize the summarizer with a smaller, faster model
        logger.info("🤖 Loading summarization model...")
        try:
            self.summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",  # Smaller, faster model
                device=-1  # Use CPU to avoid GPU memory issues
            )
            logger.info("✅ Summarizer loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load summarizer: {e}")
            self.summarizer = None
    
    def search_place(self, query: str) -> Dict:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_maps",
            "q": query,
            "api_key": self.serpapi_key
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    logger.error(f"❌ HTTP Error {response.status_code}: {response.text}")
                    return {"places": [], "error": f"HTTP {response.status_code}"}
                
                data = response.json()
                
                if "error" in data:
                    logger.error(f"❌ API Error: {data['error']}")
                    return {"places": [], "error": data["error"]}
                
                places = []
                if "local_results" in data:
                    for place in data["local_results"]:
                        places.append({
                            "title": place.get("title", ""),
                            "place_id": place.get("place_id", ""),
                            "rating": place.get("rating", 0),
                            "reviews": place.get("reviews", 0),
                            "address": place.get("address", ""),
                            "type": place.get("type", "")
                        })
                
                logger.info(f"✓ Found {len(places)} places")
                return {"places": places, "search_metadata": data.get("search_metadata", {})}
                
            except requests.RequestException as e:
                logger.error(f"❌ Network error searching places (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {"places": [], "error": str(e)}
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON decode error: {e}")
                return {"places": [], "error": "Invalid JSON response"}

    def get_reviews_by_place_id(self, place_id: str, max_reviews: int = 50) -> List[Dict]:
        all_reviews = []
        next_page_token = None
        pages_fetched = 0
        max_pages = max(1, (max_reviews + 7) // 8)  # Calculate pages needed (8 reviews per page)
        
        while len(all_reviews) < max_reviews and pages_fetched < max_pages:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": self.serpapi_key
            }
            
            # Add next_page_token if we have one (for subsequent pages)
            if next_page_token:
                params["next_page_token"] = next_page_token
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"📝 Fetching reviews page {pages_fetched + 1} for place ID: {place_id}")
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                    if response.status_code != 200:
                        logger.error(f"❌ HTTP Error {response.status_code}")
                        if response.status_code == 400:
                            logger.error(f"📝 Error details: {response.text}")
                        break
                    
                    try:
                        data = response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error: {e}")
                        break
                    
                    if "error" in data:
                        logger.error(f"❌ API Error: {data['error']}")
                        break
                    
                    # Process reviews from this page
                    page_reviews = []
                    if "reviews" in data:
                        logger.info(f"✅ Found {len(data['reviews'])} reviews on page {pages_fetched + 1}")
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
                    else:
                        logger.info(f"Available keys: {list(data.keys())}")
                        break
                    
                    all_reviews.extend(page_reviews)
                    pages_fetched += 1
                    
                    # Check if there are more pages
                    search_metadata = data.get("search_metadata", {})
                    next_page_token = search_metadata.get("next_page_token")
                    
                    if not next_page_token:
                        logger.info("✓ No more pages available")
                        break
                    
                    # Don't fetch more if we have enough reviews
                    if len(all_reviews) >= max_reviews:
                        break
                    
                    # Be respectful with rate limiting
                    time.sleep(2)  # Increased delay
                    break  # Success, exit retry loop
                    
                except requests.RequestException as e:
                    logger.error(f"❌ Network error fetching reviews (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        break
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # Trim to requested number of reviews
        final_reviews = all_reviews[:max_reviews]
        logger.info(f"✓ Successfully fetched {len(final_reviews)} reviews total")
        return final_reviews
    
    def get_reviews_by_query(self, query: str, num_reviews: int = 50) -> List[Dict]:
        # First, search for the place
        search_results = self.search_place(query)
        
        if "error" in search_results:
            return []
        
        if not search_results["places"]:
            logger.error(f"❌ No places found for query: {query}")
            return []
        
        # Get the first (most relevant) result
        place = search_results["places"][0]
        place_id = place["place_id"]
        
        if not place_id:
            logger.error(f"❌ No place ID found for: {place['title']}")
            return []
        
        logger.info(f"🎯 Selected place: {place['title']} ({place['rating']} ⭐, {place['reviews']} reviews)")
        logger.info(f"📍 Place ID: {place_id}")
        
        # Fetch reviews for this place
        return self.get_reviews_by_place_id(place_id, num_reviews)

    def combine_reviews_text(self, reviews: List[Dict]) -> str:
        """Combine all review texts into a single paragraph"""
        if not reviews:
            return ""
        
        # Filter out empty reviews and combine
        review_texts = [review['text'] for review in reviews if review['text'].strip()]
        combined_text = " ".join(review_texts)
        
        logger.info(f"📝 Combined {len(review_texts)} reviews into {len(combined_text)} characters")
        return combined_text
    
    def summarize_reviews(self, reviews: List[Dict]) -> str:
        """Create an objective summary using AI summarization"""
        if not reviews:
            return "No reviews to summarize"
        
        if not self.summarizer:
            return "Summarizer not available - providing basic analysis instead"
        
        combined_text = self.combine_reviews_text(reviews)
        
        if not combined_text.strip():
            return "No meaningful review content to summarize"
        
        try:
            logger.info("🤖 Generating summary...")
            
            # Limit text length to avoid model limits (typically 1024 tokens)
            max_chars = 800  # Conservative limit
            text_to_summarize = combined_text[:max_chars]
            
            if len(combined_text) > max_chars:
                text_to_summarize += "..."
                logger.info(f"Text truncated to {max_chars} characters")
            
            # Use the summarizer properly - just pass the text, not instructions
            summary_result = self.summarizer(
                text_to_summarize, 
                max_length=150, 
                min_length=30, 
                do_sample=False
            )
            
            summary = summary_result[0]['summary_text']
            
            logger.info("✅ Summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating summary: {e}")
            # Fallback to basic analysis
            return self._create_basic_summary(reviews)
    
    def _create_basic_summary(self, reviews: List[Dict]) -> str:
        """Fallback summary method using basic text analysis"""
        if not reviews:
            return "No reviews to analyze"
        
        # Calculate basic stats
        ratings = [r['rating'] for r in reviews if r['rating'] > 0]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Common positive/negative words
        positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointing', 'poor']
        
        all_text = ' '.join([r['text'].lower() for r in reviews])
        
        pos_count = sum(all_text.count(word) for word in positive_words)
        neg_count = sum(all_text.count(word) for word in negative_words)
        
        sentiment = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
        
        return f"Based on {len(reviews)} reviews, this business has an average rating of {avg_rating:.1f}/5 with generally {sentiment} customer feedback. Reviews mention various aspects of service, quality, and customer experience."

    def save_reviews_to_csv(self, reviews: List[Dict], filename: str = None) -> str:
        """Save reviews to a CSV file"""
        if not reviews:
            logger.error("❌ No reviews to save")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reviews_{timestamp}.csv"
        
        try:
            df = pd.DataFrame(reviews)
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"✅ Reviews saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Error saving reviews: {e}")
            return ""

def main():
    # Check environment variable first
    serp_key = os.getenv("SERPAPI")
    
    # If no environment variable, prompt for manual input
    if not serp_key:
        print("❌ SERPAPI environment variable not found!")
        print("💡 Please enter your SerpAPI key manually:")
        serp_key = input("Enter your SerpAPI key: ").strip()
        
        if not serp_key:
            print("❌ No API key provided. Exiting.")
            return
    
    # Don't print API key for security
    print("🔑 API key loaded successfully")
    
    # Initialize analyzer
    analyzer = GoogleReviewsAnalyzer(serp_key)
    
    # Get search query from user input instead of hardcoding
    query = input("Enter business name to search for: ").strip()
    if not query:
        query = "Wheel coffee roasters"  # Default fallback
    
    try:
        num_reviews = int(input("Number of reviews to fetch (default 10): ") or "10")
    except ValueError:
        num_reviews = 10
    
    print(f"\n🎯 Searching for: {query}")
    print(f"📝 Fetching {num_reviews} reviews...")
    
    # Get reviews
    reviews = analyzer.get_reviews_by_query(query, num_reviews)
    
    if reviews:
        print(f"\n✅ Found {len(reviews)} reviews!")
        
        # Display all reviews
        print("\n📝 All reviews:")
        for i, review in enumerate(reviews, 1):
            print(f"\n{i}. {review['author']} ({review['rating']}⭐)")
            print(f"   Date: {review['date']}")
            print(f"   Text: {review['text']}")
        
        # Generate summary
        print(f"\n{'='*50}")
        print("📄 REVIEW SUMMARY")
        print(f"{'='*50}")
        
        summary = analyzer.summarize_reviews(reviews)
        print(f"\n{summary}")
        
        # Simple sentiment analysis
        positive_reviews = [r for r in reviews if r['rating'] >= 4]
        negative_reviews = [r for r in reviews if r['rating'] <= 2]
        
        print(f"\n📊 Quick Analysis:")
        print(f"   Average Rating: {sum(r['rating'] for r in reviews) / len(reviews):.1f}/5")
        print(f"   Positive (4-5 stars): {len(positive_reviews)}/{len(reviews)} ({len(positive_reviews)/len(reviews)*100:.1f}%)")
        print(f"   Negative (1-2 stars): {len(negative_reviews)}/{len(reviews)} ({len(negative_reviews)/len(reviews)*100:.1f}%)")
        
        # Ask if user wants to save to CSV
        save_choice = input("\nSave reviews to CSV? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = analyzer.save_reviews_to_csv(reviews)
            if filename:
                print(f"✅ Reviews saved to: {filename}")
        
    else:
        print("\n❌ No reviews found.")
        print("💡 This might be because:")
        print("   1. The place has no reviews")
        print("   2. Reviews are not publicly accessible")
        print("   3. Temporary API issues")
        print("   4. Invalid API key")

if __name__ == "__main__":
=======
import requests
import json
import time
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import os
from transformers import pipeline

class GoogleReviewsAnalyzer:
    
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        print(f"Initialized with SerpAPI")
        
        # Initialize the summarizer
        print("🤖 Loading summarization model...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("✅ Summarizer loaded successfully")
    
    def search_place(self, query: str) -> Dict:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_maps",
            "q": query,
            "api_key": self.serpapi_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return {"places": [], "error": f"HTTP {response.status_code}"}
            
            data = response.json()
            
            if "error" in data:
                print(f"❌ API Error: {data['error']}")
                return {"places": [], "error": data["error"]}
            
            places = []
            if "local_results" in data:
                for place in data["local_results"]:
                    places.append({
                        "title": place.get("title", ""),
                        "place_id": place.get("place_id", ""),
                        "rating": place.get("rating", 0),
                        "reviews": place.get("reviews", 0),
                        "address": place.get("address", ""),
                        "type": place.get("type", "")
                    })
            
            print(f"✓ Found {len(places)} places")
            return {"places": places, "search_metadata": data.get("search_metadata", {})}
            
        except requests.RequestException as e:
            print(f"❌ Network error searching places: {e}")
            return {"places": [], "error": str(e)}
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return {"places": [], "error": "Invalid JSON response"}

    def get_reviews_by_place_id(self, place_id: str, max_reviews: int = 50) -> List[Dict]:
        all_reviews = []
        next_page_token = None
        pages_fetched = 0
        max_pages = max(1, (max_reviews + 7) // 8)  # Calculate pages needed (8 reviews per page)
        
        while len(all_reviews) < max_reviews and pages_fetched < max_pages:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": self.serpapi_key
            }
            
            # Add next_page_token if we have one (for subsequent pages)
            if next_page_token:
                params["next_page_token"] = next_page_token
            
            try:
                print(f"📝 Fetching reviews page {pages_fetched + 1} for place ID: {place_id}")
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"❌ HTTP Error {response.status_code}")
                    if response.status_code == 400:
                        print(f"📝 Error details: {response.text}")
                    break
                
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    break
                
                if "error" in data:
                    print(f"❌ API Error: {data['error']}")
                    break
                
                # Process reviews from this page
                page_reviews = []
                if "reviews" in data:
                    print(f"✅ Found {len(data['reviews'])} reviews on page {pages_fetched + 1}")
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
                else:
                    print(f"Available keys: {list(data.keys())}")
                    break
                
                all_reviews.extend(page_reviews)
                pages_fetched += 1
                
                # Check if there are more pages
                search_metadata = data.get("search_metadata", {})
                next_page_token = search_metadata.get("next_page_token")
                
                if not next_page_token:
                    print("✓ No more pages available")
                    break
                
                # Don't fetch more if we have enough reviews
                if len(all_reviews) >= max_reviews:
                    break
                
                # Be respectful with rate limiting
                time.sleep(1)
                
            except requests.RequestException as e:
                print(f"❌ Network error fetching reviews: {e}")
                break
        
        # Trim to requested number of reviews
        final_reviews = all_reviews[:max_reviews]
        print(f"✓ Successfully fetched {len(final_reviews)} reviews total")
        return final_reviews
    
    def get_reviews_by_query(self, query: str, num_reviews: int = 50) -> List[Dict]:
        # First, search for the place
        search_results = self.search_place(query)
        
        if "error" in search_results:
            return []
        
        if not search_results["places"]:
            print(f"❌ No places found for query: {query}")
            return []
        
        # Get the first (most relevant) result
        place = search_results["places"][0]
        place_id = place["place_id"]
        
        if not place_id:
            print(f"❌ No place ID found for: {place['title']}")
            return []
        
        print(f"🎯 Selected place: {place['title']} ({place['rating']} ⭐, {place['reviews']} reviews)")
        print(f"📍 Place ID: {place_id}")
        
        # Fetch reviews for this place
        return self.get_reviews_by_place_id(place_id, num_reviews)

    def combine_reviews_text(self, reviews: List[Dict]) -> str:
        """Combine all review texts into a single paragraph"""
        if not reviews:
            return ""
        
        # Filter out empty reviews and combine
        review_texts = [review['text'] for review in reviews if review['text'].strip()]
        combined_text = " ".join(review_texts)
        
        print(f"📝 Combined {len(review_texts)} reviews into {len(combined_text)} characters")
        return combined_text
    
    def summarize_reviews(self, reviews: List[Dict]) -> str:
        """Create an objective summary by extracting key themes"""
        if not reviews:
            return "No reviews to summarize"
        
        combined_text = self.combine_reviews_text(reviews)
        
        if not combined_text.strip():
            return "No meaningful review content to summarize"
        
        try:
            print("🤖 Generating summary...")
            
            # Create a more structured prompt for objective summary
            prompt = f"""Based on these customer reviews, provide an objective summary of the business:

    Reviews: {combined_text[:3000]}

    Focus on: quality, service, products, customer experience, and overall reputation."""
            
            summary_result = self.summarizer(prompt, max_length=200, min_length=50, do_sample=False)
            summary = summary_result[0]['summary_text']
            
            print("✅ Summary generated successfully")
            return summary
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

    def save_reviews_to_csv(self, reviews: List[Dict], filename: str = None) -> str:
        """Save reviews to a CSV file"""
        if not reviews:
            print("❌ No reviews to save")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reviews_{timestamp}.csv"
        
        try:
            df = pd.DataFrame(reviews)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"✅ Reviews saved to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving reviews: {e}")
            return ""

def main():
    # Check environment variable first
    serp_key = os.getenv("SERPAPI")
    
    # If no environment variable, prompt for manual input
    if not serp_key:
        print("❌ SERPAPI environment variable not found!")
        print("💡 Please enter your SerpAPI key manually:")
        serp_key = input("Enter your SerpAPI key: ").strip()
        
        if not serp_key:
            print("❌ No API key provided. Exiting.")
            return
    
    print(f"🔑 Using API key: {serp_key[:3]}")
    
    # Initialize analyzer
    analyzer = GoogleReviewsAnalyzer(serp_key)
    
    # Get search query
    query = "Wheel coffee roasters"
    num_reviews = 10  # Changed to 10 as requested
    
    print(f"\n🎯 Searching for: {query}")
    print(f"📝 Fetching {num_reviews} reviews...")
    
    # Get reviews
    reviews = analyzer.get_reviews_by_query(query, num_reviews)
    
    if reviews:
        print(f"\n✅ Found {len(reviews)} reviews!")
        
        # Display all reviews
        print("\n📝 All reviews:")
        for i, review in enumerate(reviews, 1):
            print(f"\n{i}. {review['author']} ({review['rating']}⭐)")
            print(f"   Date: {review['date']}")
            print(f"   Text: {review['text']}")
        
        # Generate summary
        print(f"\n{'='*50}")
        print("📄 REVIEW SUMMARY")
        print(f"{'='*50}")
        
        summary = analyzer.summarize_reviews(reviews)
        print(f"\n{summary}")
        
        # Simple sentiment analysis
        positive_reviews = [r for r in reviews if r['rating'] >= 4]
        negative_reviews = [r for r in reviews if r['rating'] <= 2]
        
        print(f"\n📊 Quick Analysis:")
        print(f"   Average Rating: {sum(r['rating'] for r in reviews) / len(reviews):.1f}/5")
        print(f"   Positive (4-5 stars): {len(positive_reviews)}/{len(reviews)} ({len(positive_reviews)/len(reviews)*100:.1f}%)")
        print(f"   Negative (1-2 stars): {len(negative_reviews)}/{len(reviews)} ({len(negative_reviews)/len(reviews)*100:.1f}%)")
        
    else:
        print("\n❌ No reviews found.")
        print("💡 This might be because:")
        print("   1. The place has no reviews")
        print("   2. Reviews are not publicly accessible")
        print("   3. Temporary API issues")

if __name__ == "__main__":
>>>>>>> 980a57f5f1c976cafaedfd4cff04c29d00369cb0
    main()