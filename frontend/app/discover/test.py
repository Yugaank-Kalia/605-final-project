import requests
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://7626-18-219-93-201.ngrok-free.app/api/recommend"  # Replace with your actual endpoint
payload = {
    "spotify_url": "https://open.spotify.com/track/6AI3ezQ4o3HUoP6Dhudph3?si=b7fa4ced8f9c48d0"  # Example Spotify track URL
}

def hit_endpoint():
    try:
        response = requests.post(API_URL, json=payload)
        return response.status_code
    except Exception as e:
        return f"Error: {e}"

# Run 100 parallel POST requests
with ThreadPoolExecutor(max_workers=100) as executor:
    results = list(executor.map(lambda _: hit_endpoint(), range(100)))

print(results)