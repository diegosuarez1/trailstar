from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import structlog
import os
from dotenv import load_dotenv
import math

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = structlog.get_logger()

app = Flask(__name__)
CORS(app)

# Anthropic API setup - now using environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth using the Haversine formula"""
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def query_claude_api(messages, max_tokens=150):
    """Call Anthropic Claude API"""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Convert messages to Claude format
    system_message = ""
    claude_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            claude_messages.append({
                "role": msg["role"], 
                "content": msg["content"]
            })
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": claude_messages
    }
    
    if system_message:
        payload["system"] = system_message
    
    try:
        logger.info("calling_claude_api", prompt_length=len(str(claude_messages)))
        
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 401:
            logger.error("claude_api_unauthorized", error="Invalid API key")
            return "Error: Invalid API key"
        elif response.status_code == 403:
            logger.error("claude_api_forbidden", error="Access denied")
            return "Error: Access denied"
        elif response.status_code == 429:
            logger.error("claude_api_rate_limited", error="Rate limited")
            return "Error: Rate limited"
        
        response.raise_for_status()
        
        result = response.json()
        logger.info("claude_api_raw_response", response=str(result)[:200])
        
        if 'content' in result and len(result['content']) > 0:
            generated_text = result['content'][0]['text']
            logger.info("claude_api_success", response_length=len(generated_text))
            return generated_text.strip()
        else:
            logger.error("unexpected_claude_response", response=result)
            return "Error: Unexpected API response"
            
    except requests.exceptions.Timeout:
        logger.error("claude_api_timeout")
        return "Error: API timeout"
    except Exception as e:
        logger.error("claude_api_error", error=str(e))
        return f"Error: {str(e)}"

def extract_coordinates_from_geometry(geometry_string):
    """Extract coordinates from LINESTRING geometry"""
    try:
        coords_part = geometry_string.replace('LINESTRING (', '').replace(')', '')
        first_point = coords_part.split(',')[0].strip()
        lon, lat = first_point.split()
        return [float(lat), float(lon)]
    except:
        return [37.2431, -80.4139]

def classify_difficulty_new_format(trail):
    """Classify trail difficulty based on new format features"""
    max_slope = trail['slope_max']
    length = trail['trail_length (km)']
    percent_steep = trail['percent_steep']
    
    if max_slope > 15 or (length > 10 and percent_steep > 20):
        return 'hard'
    elif max_slope > 8 or (length > 5 and percent_steep > 10):
        return 'moderate'
    else:
        return 'easy'

def generate_basic_description_new_format(trail):
    """Generate a simple basic description for new format without AI"""
    difficulty = classify_difficulty_new_format(trail)
    length = round(trail['trail_length (km)'], 1)
    
    return f"A {difficulty} {length}km trail."

def load_trails():
    """Load trails from trails_100.json file"""
    try:
        with open('trails_100.json', 'r') as f:
            trails_data = json.load(f)
        
        processed_trails = []
        for i, trail in enumerate(trails_data):
            processed_trail = {
                'id': f'trail_{i:03d}',
                'name': trail['trail_name'],
                'length': round(trail['trail_length (km)'], 1),
                'difficulty': classify_difficulty_new_format(trail),
                'surface': 'dirt',
                'vehicles': 'biking' if trail['bikes_allowed'] == 1 else 'hiking',
                'description': generate_basic_description_new_format(trail),
                'coordinates': extract_coordinates_from_geometry(trail['geometry']),
                'elevation_profile': [int(x) for x in trail['elevation_profile']],
                'max_slope': round(trail['slope_max'], 1),
                'avg_slope': round(trail['average_slope'], 1),
                'distance_to_water': round(trail['distance_to_water'], 1),
                'raw_trail_data': trail
            }
            processed_trails.append(processed_trail)
        
        return processed_trails
    except FileNotFoundError:
        logger.error("trails_100.json not found")
        return []

# Load trails at startup
TRAILS_DATA = load_trails()

def passes_filters(trail, filters, user_location=None):
    """Check if trail passes the specified filters"""
    if filters.get('length') and trail['length'] > filters['length']:
        return False
    if filters.get('difficulty') and trail['difficulty'] != filters['difficulty']:
        return False
    if filters.get('surface') and trail['surface'] != filters['surface']:
        return False
    if filters.get('vehicles'):
        if filters['vehicles'] == 'hiking' and trail['vehicles'] != 'hiking':
            return False
        elif filters['vehicles'] == 'biking' and trail['vehicles'] != 'biking':
            return False
    
    # Location filter
    if filters.get('max_distance') and user_location:
        trail_distance = haversine_distance(
            user_location['lat'], user_location['lon'],
            trail['coordinates'][0], trail['coordinates'][1]
        )
        if trail_distance > filters['max_distance']:
            return False
            
    return True

def filter_trails_basic(filters, user_location=None):
    """Basic filtering without AI"""
    filtered = [trail for trail in TRAILS_DATA if passes_filters(trail, filters, user_location)]
    
    # If user location is provided, sort by distance
    if user_location:
        for trail in filtered:
            trail['distance_from_user'] = haversine_distance(
                user_location['lat'], user_location['lon'],
                trail['coordinates'][0], trail['coordinates'][1]
            )
        filtered.sort(key=lambda x: x.get('distance_from_user', float('inf')))
    
    return filtered[:5]

def get_keyword_trail_recommendations(user_query, trails):
    """Simple keyword-based trail matching"""
    query_lower = user_query.lower()
    scored_trails = []
    
    for trail in trails:
        score = 0
        trail_text = f"{trail['name']} {trail['description']}".lower()
        
        if 'easy' in query_lower and trail['difficulty'] == 'easy':
            score += 10
        elif 'moderate' in query_lower and trail['difficulty'] == 'moderate':
            score += 10
        elif 'hard' in query_lower and trail['difficulty'] == 'hard':
            score += 10
        elif 'challenging' in query_lower and trail['difficulty'] == 'hard':
            score += 8
        
        if 'short' in query_lower and trail['length'] < 5:
            score += 5
        elif 'long' in query_lower and trail['length'] > 10:
            score += 5
        
        if 'bike' in query_lower and trail['vehicles'] == 'biking':
            score += 8
        if 'hike' in query_lower and trail['vehicles'] == 'hiking':
            score += 3
        
        # If user location is available, prioritize closer trails slightly
        if hasattr(trail, 'distance_from_user'):
            if trail['distance_from_user'] < 10:  # Within 10km
                score += 3
            elif trail['distance_from_user'] < 25:  # Within 25km
                score += 1
        
        scored_trails.append((score, trail))
    
    scored_trails.sort(key=lambda x: x[0], reverse=True)
    return [trail for score, trail in scored_trails[:5]]

def get_claude_trail_recommendations(user_query, filters, user_location=None):
    """Use Claude to rank trails based on user query with keyword fallback"""
    
    filtered_trails = [trail for trail in TRAILS_DATA if passes_filters(trail, filters, user_location)]
    if not filtered_trails:
        filtered_trails = TRAILS_DATA
    
    # Add distance information if user location is available
    if user_location:
        for trail in filtered_trails:
            trail['distance_from_user'] = haversine_distance(
                user_location['lat'], user_location['lon'],
                trail['coordinates'][0], trail['coordinates'][1]
            )
        filtered_trails.sort(key=lambda x: x.get('distance_from_user', float('inf')))
    
    try:
        trail_descriptions = []
        for trail in filtered_trails[:20]:
            distance_info = ""
            if hasattr(trail, 'distance_from_user'):
                distance_info = f" ({round(trail['distance_from_user'], 1)}km away)"
            desc = f"ID {trail['id']}: {trail['name']} - {trail['length']}km, {trail['difficulty']} difficulty, {trail['vehicles']}{distance_info}"
            trail_descriptions.append(desc)
        
        location_context = ""
        if user_location:
            location_context = " Consider distance from user when ranking - closer trails are generally preferable."
        
        messages = [
            {
                "role": "system",
                "content": f"You are a trail recommendation expert. Analyze user requests and recommend trails by ranking them from most to least suitable.{location_context} Respond with ONLY a comma-separated list of trail IDs in order of recommendation."
            },
            {
                "role": "user",
                "content": f"""User Request: "{user_query}"

Available Trails:
{chr(10).join(trail_descriptions)}

Rank these trails from best match to worst match for this user. Respond with only the trail IDs separated by commas."""
            }
        ]
        
        logger.info("attempting_ai_recommendation")
        response = query_claude_api(messages, max_tokens=100)
        
        if not response.startswith('Error:'):
            import re
            trail_ids = re.findall(r'trail_\d+', response)
            
            if trail_ids:
                recommended = []
                for trail_id in trail_ids[:5]:
                    trail = next((t for t in filtered_trails if t['id'] == trail_id), None)
                    if trail:
                        recommended.append(trail.copy())
                
                if len(recommended) >= 3:
                    if len(recommended) < 5:
                        remaining = [t for t in filtered_trails if t['id'] not in trail_ids]
                        recommended.extend(remaining[:5-len(recommended)])
                    
                    logger.info("ai_recommendation_success", count=len(recommended))
                    return recommended
        
        logger.info("ai_failed_using_keyword_fallback", response=response[:100])
        return get_keyword_trail_recommendations(user_query, filtered_trails)
        
    except Exception as e:
        logger.info("using_keyword_fallback", error=str(e))
        return get_keyword_trail_recommendations(user_query, filtered_trails)

def generate_trail_summary_claude(trail):
    """Generate engaging trail summary using Claude"""
    distance_info = ""
    if hasattr(trail, 'distance_from_user'):
        distance_info = f" Located {round(trail['distance_from_user'], 1)}km from your location."
    
    messages = [
        {
            "role": "system",
            "content": "You are an enthusiastic trail guide who writes engaging, concise descriptions. Write 1-2 sentences that capture what makes each trail special."
        },
        {
            "role": "user",
            "content": f"""Trail: {trail['name']}
Length: {trail['length']} km
Difficulty: {trail['difficulty']}
Max Slope: {trail.get('max_slope', 'N/A')}%{distance_info}

Write an engaging 1-2 sentence summary highlighting what makes this trail unique and appealing to hikers."""
        }
    ]
    
    try:
        summary = query_claude_api(messages, max_tokens=80)
        return summary if summary and len(summary) > 10 and not summary.startswith('Error:') else trail['description']
    except:
        return trail['description']

@app.route('/api/search', methods=['POST'])
def search_trails():
    try:
        data = request.get_json()
        query = data.get('query', '')
        filters = data.get('filters', {})
        user_location = data.get('user_location')
        
        logger.info("trail_search_request", query=query, filters=filters, has_location=bool(user_location))
        
        if query.strip():
            recommended_trails = get_claude_trail_recommendations(query, filters, user_location)
        else:
            recommended_trails = filter_trails_basic(filters, user_location)
        
        for trail in recommended_trails:
            if query.strip():
                trail['ai_summary'] = generate_trail_summary_claude(trail)
            else:
                trail['ai_summary'] = trail['description']
        
        logger.info("trail_search_complete", results_count=len(recommended_trails))
        return jsonify({
            'trails': recommended_trails,
            'used_ai': bool(query.strip())
        })
        
    except Exception as e:
        logger.error("search_api_error", error=str(e))
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/trails', methods=['GET'])
def get_all_trails():
    """Get all trails for initial display"""
    return jsonify({'trails': TRAILS_DATA[:5]})

@app.route('/')
def index():
    """Serve the main HTML page"""
    with open('index.html', 'r') as f:
        return f.read()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)