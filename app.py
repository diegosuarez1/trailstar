# app.py
import os
import json
import math
import structlog
import asyncio
import httpx
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Logging
logger = structlog.get_logger()

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")


# --- Utilities ---
def extract_trailhead(geometry_string):
    try:
        if geometry_string.startswith("LINESTRING"):
            coords_part = geometry_string.replace('LINESTRING (', '').replace(')', '')
            first_point = coords_part.split(',')[0].strip()
            lon, lat = first_point.split()
            return [float(lat), float(lon)]
        elif geometry_string.startswith("MULTILINESTRING"):
            coords_part = geometry_string.replace('MULTILINESTRING ((', '').replace('))', '')
            first_point = coords_part.split(',')[0].strip()
            lon, lat = first_point.split()
            return [float(lat), float(lon)]
        else:
            logger.error("Unknown geometry type", geom=geometry_string)
            return [0, 0]
    except Exception as e:
        logger.error("extract_trailhead_error", error=str(e))
        return [0, 0]

def load_trails():
    try:
        with open('trails_full.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("trails not found")
        return []

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r

# --- Claude API call ---
async def query_claude_api(messages, max_tokens=150):
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

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

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        if 'content' in result and len(result['content']) > 0:
            return result['content'][0]['text'].strip()
        return "Error: No content"

# --- Generate trail descriptions ---
async def generate_desc(trail):
    closest_natural_str = ", ".join(
        [f"{name} ({dist} km)" for name, dist in trail.get('closest_natural', [])]
    ) or "N/A"

    closest_cities_str = ", ".join(
        [f"{name} ({dist} km)" for name, dist in trail.get('closest_cities', [])]
    ) or "N/A"

    messages = [
        {
        "role": "system",
        "content": (
            "You are a hiking guide expert. "
            "Generate engaging, concise 1-2 sentence trail summaries written in natural human language. "
            "Highlight key attributes such as difficulty, length, elevation gain, slope, remoteness, terrain, natural features or landmarks."
            "Use descriptive, relatable phrases that hikers would naturally use"
            "Avoid repeating raw data; integrate the information smoothly into a readable description."
        )
    },
        {
            "role": "user",
            "content": f"""Trail: {trail.get('trail_name')}
Length: {trail.get('trail_length (km)', 'N/A')} km
Difficulty: {trail.get('difficulty')}
Net Gain: {trail.get('net_gain', 'N/A')} m
Max Slope: {trail.get('slope_max', 'N/A')}%
Average Slope: {trail.get('average_slope', 'N/A')}%
Max Elevation: {trail.get('max_elevation', 'N/A')}
Curvature: {trail.get('curvature', 'N/A')}
Distance to water: {trail.get('distance_to_water', 'N/A')}
Distance to city: {trail.get('distance_to_city', 'N/A')}
Closest natural features: {closest_natural_str}
Closest cities: {closest_cities_str}

Write a 1-2 sentence engaging summary emphasizing what makes this trail appealing and unique for hikers."""        }
    ]

    try:
        return await query_claude_api(messages)
    except Exception as e:
        logger.error("generate_desc_error", error=str(e))
        return "Error generating description"


# --- Load trails on startup ---
trails = load_trails()
trails_df = pd.DataFrame(trails)
trails_df['coordinates'] = trails_df['geometry'].apply(extract_trailhead)
print("Successfully loaded trails")


# --- API endpoint ---
@app.post("/api/search")
async def search_trails(request: Request):
    try:
        data = await request.json()
        query = data.get('query', '')
        filters = data.get('filters', {})
        #difficulty = "Hard"
        user_location = data.get('user_location')

        if not user_location:
            return {"error": "user_location required"}

        user_lat = float(user_location['lat'])
        user_long = float(user_location['lon'])

        # Filter trails
        filtered_df = trails_df.copy()
        filtered_df['distance_from_user'] = filtered_df.apply(
            lambda row: float(haversine_distance(user_lat, user_long, row['coordinates'][0], row['coordinates'][1])),
            axis=1
        )
        filtered_df = filtered_df[
            (filtered_df["trail_length (km)"] <= filters.get('length', float('inf'))) &
            (filtered_df['distance_from_user'] <= filters.get('max_distance', float('inf')))
        ]
        ordered_df = filtered_df.sort_values(by='distance_from_user', ascending=True)
        ordered_df['original_index'] = ordered_df.index

        trails_list = ordered_df.to_dict(orient="records")
        
        logger.info("Found suitable trails", count=len(trails_list))

        # Generate descriptions concurrently
        summaries = await asyncio.gather(*(generate_desc(trail) for trail in trails_list))
        embeddings = model.encode(summaries, convert_to_tensor=True)
        input_emb = model.encode(query, convert_to_tensor=True)
        similarity_scores = [util.cos_sim(input_emb, emb) for emb in embeddings]
        
        top_n = 5

        results_df = pd.DataFrame(trails_list)
        results_df['description'] = summaries
        results_df['similarity_score'] = [score.item() for score in similarity_scores]
        results_df = results_df.sort_values('similarity_score', ascending=False)

        from IPython.display import display
        
        results_df = results_df.iloc[:top_n]
        print(results_df.columns)
        display(results_df)  

        results_dict = results_df.to_dict(orient="records")
            
        return {
            "trails": results_dict,  # Limit results
            "used_ai": bool(query.strip())
        }

    except Exception as e:
        logger.error("search_api_error", error=str(e))
        return {"error": "Search failed"}


# --- Serve index.html ---
from fastapi.responses import FileResponse

@app.get("/")
async def index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)