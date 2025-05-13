import os
import random
import re
import json
import openai
import spotipy
import numpy as np
import pandas as pd
from flask_cors import CORS
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials
# from sklearn.discriminant_analysis import StandardScaler
import boto3
from io import StringIO
from sklearn.preprocessing import StandardScaler  # replaces discriminant_analysis version (why do we do this?)

load_dotenv()

app = Flask(__name__)
CORS(app)

# Environment variables
client_id = os.getenv("SPOTIFY_CLIENT_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Spotify API setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

class KNNRecommender:
    def __init__(self, recommendations=10):
        self.exit_loop = False  # a flag for exiting recursive calls in get_spot_recommendations function
        self.scaler = StandardScaler()
        self.recommendations = recommendations  # number of recommendations given for a song
        self.main_kaggle_genres = ["edm", "rap", "pop", "r&b", "latin", "rock"]
        # S3 read logic
        s3 = boto3.client('s3')
        obj = s3.get_object(
            Bucket='my-spotify-song-csv-bucket',
            Key='spotify_songs_cleaned.csv'
        )
        self.df = pd.read_csv(obj['Body'])
        print("Loaded dataset from S3.")
        self.features =['danceability', 'energy', 'key', 'loudness', 'mode','speechiness', 'acousticness', 'instrumentalness','liveness', 'valence', 'tempo']
        self.X_scaled = self.scaler.fit_transform(self.df[self.features])
    
    # extracts Spotify ID from the given Spotify URL
    @staticmethod
    def extract_track_id(url):
        match = re.search(r"track/([a-zA-Z0-9]+)", url)
        return match.group(1) if match else None

    # uses OpenAI to find a song genre if not given in dataset or Spotify API
    @staticmethod
    def obtain_genre(song_title, artist_name, release_year):
        prompt = f"""
        Predict plausible Spotify audio feature values for a song given its metadata.

        Song title: {song_title}
        Artist: {artist_name}
        Release year: {release_year}

        Return a JSON object with the following fields and valid ranges:
        - genre (choose a single genre from: "edm", "pop", "rock", "r&b", "rap", "latin". If the song's actual genre is not listed, search google and provide the actual genre)

        Example output format:
        {{
            "genre": "rap",
        }}
        Do not include any markdown code blocks (no triple backticks). Only return raw JSON.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # or "gpt-3.5-turbo" if you prefer
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )

            generated_text = response['choices'][0]['message']['content'].strip()

            # Parse the JSON output safely
            features = json.loads(generated_text)

            # Return the features as a Python dictionary
            return features

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return None
    
    # returns the arist's name, track name, artist genres (as listed in Spotify API), and release year for a certain track
    def get_spotify_song_info(self, track_id):
        try:
            track = sp.track(track_id)
            artist_id = track['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            track_name = track['name']
            artist_name = track['artists'][0]['name']
            artist_genres = artist_info['genres']
            release_year = int(track['album']['release_date'][:4])
            if not artist_genres:
                print("Original artist has no spotify genre info...obtaining info from openai")
                estimated_genre = self.obtain_genre(track_name, artist_name, release_year)
                genre = [estimated_genre.get("genre")]
                print(f"estimated genre: {genre}")
                return artist_name, track_name, genre, release_year
            print(f"spotify genres: {artist_genres}")
            return artist_name, track_name, artist_genres, release_year
        except:
            return None, None, None, None

    # sets up our KNN model which will be used to find similar songs based on certain features
    def getNearestNeighbors(self):
        knn = NearestNeighbors(n_neighbors=self.recommendations + 1, algorithm='auto')
        knn.fit(self.X_scaled)
        return knn

    # returns the necessary info for the front end for an input song (returns track name, artist name, url, imageURL, genre(s), and release date)
    @staticmethod
    def get_spotify_info(track_name, artist_name):
        result = sp.search(q=f"track:{track_name} artist:{artist_name}", type="track", limit=1)
        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            artist_id = track['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            artist_genres = artist_info['genres']
            release_year = int(track['album']['release_date'][:4])
            return {
                'title': track['name'],
                'artist': track['artists'][0]['name'],
                'url': f'https://open.spotify.com/track/{track['id']}',
                'imageUrl': track['album']['images'][0]['url'],
                "genre": artist_genres,
                "release_date": release_year
            }
        return None

    # utilizes the Spotify API to recommend songs given genre, release year, and related artists
    def get_spot_recommendations(self, artist_name, track_name, track_id, songs_to_return=None, genre_sample_size=None, num_related_artists=None, tracks_per_artist=2, retry=False, genre=None):
        if not genre_sample_size:
            genre_sample_size = self.recommendations
        if not num_related_artists:
            num_related_artists = self.recommendations//2
        if not songs_to_return:
            songs_to_return = self.recommendations

        # Step 1: Search for the track
        search_results = sp.search(q=artist_name, type='artist', limit=1)
        if search_results:
            artist = search_results['artists']['items'][0]
            artist_id = artist['id']
        else:
            print("Artist not found in spotify...")
            return None
        try:
            track = sp.track(track_id)
            print(f"Found track: {track['name']} by {track['artists'][0]['name']}")
        except spotipy.exceptions.SpotifyException as e:
            print(f"Error fetching track with ID {track_id}: {e}")
            return None

        # Step 4: Get release year of the original track
        original_release_date = track['album']['release_date']
        original_release_year = int(original_release_date[:4])
        min_year = original_release_year - 3
        max_year = original_release_year + 3
        print(f"Original release year: {original_release_year}. Searching for tracks from {min_year} to {max_year}.")
        
        # Step 2: Get the artist's genres
        if retry or genre is not None:
            print("we are retrying search with openai provided genre")
            artist_genres = genre
            print(f"estimated genre: {genre}")
            if genre in self.main_kaggle_genres:
                return self.recommend_songs(track_id, songs_to_return, artist_name=artist_name, song_name=track_name, spotify_genres=artist_genres, spotify_release_year=original_release_year)
        else:
            artist_info = sp.artist(artist_id)
            artist_genres = set(artist_info['genres'])
            if not artist_genres:
                print("Original artist has no genre info.")
                return []

        # # Step 3: Get all genres from the artist
        if isinstance(artist_genres, (list, set)):
            selected_genres = artist_genres
        else:
            selected_genres = [artist_genres]
        print(f"Selected genres: {selected_genres}")

        
        original_track_id = track['id']  # Save original track ID

        track_dict = {}  # key = track ID, value = track info

        # ---- Part 1: Get tracks by genre ----
        for genre in selected_genres:
            genre_query = f'genre:"{genre}"'
            for offset in [0, 50, 100, 150]:  # Up to 200 results
                results = sp.search(q=genre_query, type='track', limit=50, offset=offset)['tracks']['items']
                for t in results:
                    release_date = t['album'].get('release_date')
                    if release_date:
                        try:
                            release_year = int(release_date[:4])
                            if min_year <= release_year <= max_year:
                                if t['id'] != original_track_id and t['id'] not in track_dict:
                                    track_dict[t['id']] = {
                                        'track_name': t['name'],
                                        'track_artist': t['artists'][0]['name'],
                                        'release_date': release_date,
                                        'source': 'genre'
                                    }
                        except ValueError:
                            continue

        # ---- Part 2: Get tracks from related artists with overlapping genres and year filtering ----
        try:
            related_artists_response = sp.artist_related_artists(artist_id)
            related_artists = related_artists_response.get('artists', [])

            # Filter related artists by shared genres
            filtered_artists = [a for a in related_artists if artist_genres & set(a['genres'])]
            filtered_artists = filtered_artists[:num_related_artists]
        except SpotifyException as e:
            if e.http_status == 404:
                print(f"Artist ID {artist_id} has no related artists or is invalid. Skipping related artists.")
                filtered_artists = []  # Proceed with genre-based recommendations only
            else:
                raise  # Re-raise for other HTTP errors
        filtered_artists = []  # Skip related artist step

        for artist in filtered_artists:
            top_tracks = sp.artist_top_tracks(artist['id'], country='US')['tracks']
            for t in top_tracks[:tracks_per_artist]:
                album = sp.album(t['album']['id'])
                release_date = album.get('release_date')
                if release_date:
                    try:
                        release_year = int(release_date[:4])
                        if min_year <= release_year <= max_year:
                            if t['id'] != original_track_id and t['id'] not in track_dict:
                                track_dict[t['id']] = {
                                    'track_name': t['name'],
                                    'track_artist': artist['name'],
                                    'release_date': release_date,
                                    'source': 'related_artist'
                                }
                    except ValueError:
                        continue

        # ---- Final step: Sample and return as DataFrame ----
        if not track_dict:
            if self.exit_loop == False:
                print("no genre info...retry?")
                self.exit_loop = True
                estimated_features = self.get_estimated_audio_features(track_name, artist_name, artist_genres, original_release_year)
                print(estimated_features)
                genre = estimated_features.get("genre")
                return self.get_spot_recommendations(artist_name, track_name, retry=True, genre=genre, track_id=track_id, songs_to_return=songs_to_return)
            else:
                print(f"No tracks found for genres {selected_genres} or related artists within {min_year}-{max_year}.")
                return pd.DataFrame(columns=['track_name', 'track_artist', 'release_date', 'source'])

        all_unique_tracks = list(track_dict.values())
        sampled_tracks = random.sample(all_unique_tracks, min(genre_sample_size, len(all_unique_tracks)))

        sampled_df = pd.DataFrame(sampled_tracks, columns=[
            'track_name',
            'track_artist',
            'release_date',
            'source'
        ])

        return sampled_df

    # uses OpenAI to determine various musical and mood features for a given song, arist, and genre(s) (if provided from Spotify API)
    @staticmethod
    def get_estimated_audio_features(song_title, artist_name, genres, release_year):
        prompt = f"""
        Predict plausible Spotify audio feature values for a song given its metadata.

        Song title: {song_title}
        Artist: {artist_name}
        Genres: {', '.join(genres)}
        Release year: {release_year}

        Return a JSON object with the following fields and valid ranges:
        - genre (choose a single genre from: "edm", "pop", "rock", "r&b", "rap", "latin". If the song's actual genre is not listed, search google and provide the actual genre)
        - danceability (float between 0 and 1)
        - energy (float between 0 and 1)
        - key (integer between 0 and 11)
        - loudness (negative float, typical range -60 to 0)
        - mode (integer, 1 for major, 0 for minor)
        - speechiness (float between 0 and 1)
        - acousticness (float between 0 and 1)
        - instrumentalness (float between 0 and 1)
        - liveness (float between 0 and 1)
        - valence (float between 0 and 1)
        - tempo (float BPM, typically between 50 and 200)

        Example output format:
        {{
            "genre": "rap",
            "danceability": 0.75,
            "energy": 0.65,
            "key": 5,
            "loudness": -5.2,
            "mode": 1,
            "speechiness": 0.04,
            "acousticness": 0.15,
            "instrumentalness": 0.0,
            "liveness": 0.12,
            "valence": 0.6,
            "tempo": 120.0
        }}
        Do not include any markdown code blocks (no triple backticks). Only return raw JSON.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # or "gpt-3.5-turbo" if you prefer
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )

            generated_text = response['choices'][0]['message']['content'].strip()

            # Parse the JSON output safely
            features = json.loads(generated_text)

            # Return the features as a Python dictionary
            return features

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return None

    # Parent rcommendation function based on user input (leverages both Kaggle dataset and Spotify API, case-by-case)
    def recommend_songs(self, track_id, df_clean, songs_to_return=None, artist_name=None, song_name=None, spotify_genres=None, spotify_release_year=None):
        if songs_to_return is None:
            songs_to_return = self.recommendations

        # Use Spotify API to get genre and release year
        if artist_name is None and song_name is None and spotify_genres is None and spotify_release_year is None:
            artist_name, song_name, spotify_genres, spotify_release_year = self.get_spotify_song_info(track_id)

        if (len(spotify_genres) == 1) and (spotify_genres[0] not in self.main_kaggle_genres) or spotify_release_year > 2019 or (not any(g in self.main_kaggle_genres for g in spotify_genres)):
            return self.get_spot_recommendations(artist_name, song_name, track_id, genre=spotify_genres)

        if not spotify_genres or spotify_release_year is None:
            return None

        # Try to find the song in the dataset
        song_row_kaggle = df_clean[
            (df_clean['track_name'].str.lower() == song_name.lower()) &
            (df_clean['track_artist'].str.lower() == artist_name.lower())
        ]

        if song_row_kaggle.empty:
            print(f"\nSong '{song_name}' by '{artist_name}' not found in dataset. Using Spotify API for info...")
            print(f"\nFound song from Spotify. Genre(s): {spotify_genres}, Release Year: {spotify_release_year}")

            # Get estimated audio features from OpenAI
            estimated_features = self.get_estimated_audio_features(song_name, artist_name, spotify_genres, spotify_release_year)

            if estimated_features:
                print("Estimated audio features from OpenAI:", estimated_features)
                genre = estimated_features.get("genre")
                # Prepare the estimated features for KNN
                estimated_features_list = [estimated_features.get(f, np.nan) for f in self.features]
                if not any(np.isnan(x) for x in estimated_features_list):
                    estimated_features_df = pd.DataFrame([estimated_features_list], columns=self.features)
                    estimated_features_scaled = self.scaler.transform(estimated_features_df)
                    
                    release_year = spotify_release_year
                    relevant_df = pd.DataFrame()

                    # 1. Songs with the same Kaggle genre and subgenre within the year range
                    same_kaggle_df = df_clean[
                        (df_clean['playlist_genre'].str.lower() == genre.lower()) &
                        (df_clean['track_album_release_date'] >= release_year - 3) &
                        (df_clean['track_album_release_date'] <= release_year + 3)
                        ]
                    relevant_df = pd.concat([relevant_df, same_kaggle_df])

                    # 2. Songs with a main genre matching Spotify (within the year range)
                    if spotify_genres:
                        subgenre_matched = False
                        for spotify_genre in spotify_genres:
                            spotify_genre_df = df_clean[
                                (df_clean['playlist_subgenre'].str.lower() == spotify_genre.lower()) &  # Check subgenre first
                                (df_clean['track_album_release_date'] >= release_year - 3) &
                                (df_clean['track_album_release_date'] <= release_year + 3)
                            ]
                            if not spotify_genre_df.empty:
                                relevant_df = pd.concat([relevant_df, spotify_genre_df])
                                subgenre_matched = True
                        if not subgenre_matched: #if no subgenre matched, match the main genre
                            for spotify_genre in spotify_genres:
                                spotify_genre_df = df_clean[
                                    (df_clean['playlist_genre'].str.lower() == spotify_genre.lower()) &
                                    (df_clean['track_album_release_date'] >= release_year - 3) &
                                    (df_clean['track_album_release_date'] <= release_year + 3)
                                ]
                                relevant_df = pd.concat([relevant_df, spotify_genre_df])

                    relevant_df = relevant_df.drop_duplicates()
                    if relevant_df.empty:
                        print("Warning: No relevant songs found based on genre and year criteria...using spotify for recommendations")
                        return self.get_spot_recommendations(artist_name, song_name, track_id, self.features, songs_to_return)

                    recommendations = pd.DataFrame(columns=['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'track_album_release_date'])
                    recommended_indices = set()
                    n_neighbors = min(songs_to_return + 1, len(relevant_df))

                    while len(recommendations) < songs_to_return:
                        knn_model_openai = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
                        knn_model_openai.fit(self.scaler.transform(relevant_df[self.features])) # Use the filtered dataset

                        distances, indices = knn_model_openai.kneighbors(estimated_features_scaled)
                        closest_songs = relevant_df.iloc[indices[0][1:]] # Exclude the input itself

                        new_recommendations = closest_songs[
                            (~closest_songs.index.isin(recommended_indices)) &
                            (~closest_songs[['track_name', 'track_artist']].apply(tuple, axis=1).isin(recommendations[['track_name', 'track_artist']].apply(tuple, axis=1).tolist())) &
                            (closest_songs['track_name'].str.lower() != song_name.lower()) &
                            (closest_songs['track_artist'].str.lower() != artist_name.lower())
                        ].drop_duplicates(subset=['track_name', 'track_artist'])

                        if not new_recommendations.empty:
                            if recommendations.empty:
                                recommendations = new_recommendations.copy()
                            else:
                                recommendations = pd.concat([recommendations, new_recommendations])
                            recommended_indices.update(new_recommendations.index)

                        if len(recommendations) >= songs_to_return or n_neighbors >= len(relevant_df):
                            break
                        else:
                            n_neighbors = min(len(relevant_df), n_neighbors + 5) # Increase neighbors for next iteration

                    if not recommendations.empty:
                        return recommendations[['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'track_album_release_date']].head(songs_to_return)
                    else:
                        print("Could not find enough recommendations based on estimated audio features.")
                        return None
                else:
                    print("OpenAI did not provide all necessary audio features.")
                    return None
            else:
                print("Could not retrieve estimated audio features from OpenAI.")
                return None

        else:
            song_row = song_row_kaggle.iloc[0].copy()
            kaggle_genre = song_row['playlist_genre']
            kaggle_sub_genre = song_row['playlist_subgenre'] if pd.notna(song_row['playlist_subgenre']) else kaggle_genre
            release_year = song_row['track_album_release_date']
            print(f"\nFound '{song_name}' by '{artist_name}' in genre '{kaggle_genre}' with subgenre '{kaggle_sub_genre}', released in {release_year}.")
            print(f"Spotify Genres for '{song_name}': {spotify_genres}")

            relevant_df = pd.DataFrame()

            # 1. Songs with the same Kaggle genre and subgenre within the year range
            same_kaggle_df = df_clean[
                (df_clean['playlist_genre'].str.lower() == kaggle_genre.lower()) &
                (df_clean['playlist_subgenre'].str.lower() == kaggle_sub_genre.lower()) &
                (df_clean['track_album_release_date'] >= release_year - 3) &
                (df_clean['track_album_release_date'] <= release_year + 3)
            ]
            relevant_df = pd.concat([relevant_df, same_kaggle_df])

            # 2. Songs with a main genre matching Spotify (within the year range)
            if spotify_genres:
                subgenre_matched = False
                for spotify_genre in spotify_genres:
                    spotify_genre_df = df_clean[
                        (df_clean['playlist_subgenre'].str.lower() == spotify_genre.lower()) &  # Check subgenre first
                        (df_clean['track_album_release_date'] >= release_year - 3) &
                        (df_clean['track_album_release_date'] <= release_year + 3)
                    ]
                    if not spotify_genre_df.empty:
                        relevant_df = pd.concat([relevant_df, spotify_genre_df])
                        subgenre_matched = True
                if not subgenre_matched: #if no subgenre matched, match the main genre
                    for spotify_genre in spotify_genres:
                        spotify_genre_df = df_clean[
                            (df_clean['playlist_genre'].str.lower() == spotify_genre.lower()) &
                            (df_clean['track_album_release_date'] >= release_year - 3) &
                            (df_clean['track_album_release_date'] <= release_year + 3)
                        ]
                        relevant_df = pd.concat([relevant_df, spotify_genre_df])

            relevant_df = relevant_df.drop_duplicates()

            if relevant_df.empty:
                print("Warning: No relevant songs found based on genre and year criteria.")
                return pd.DataFrame(columns=['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'track_album_release_date'])

            recommendations = pd.DataFrame(columns=['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'track_album_release_date'])
            recommended_indices = set()
            n_neighbors = min(songs_to_return + 1, len(relevant_df))

            while len(recommendations) < songs_to_return:
                knn_model_genre_or_subgenre = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
                knn_model_genre_or_subgenre.fit(self.scaler.transform(relevant_df[self.features])) # Fit on the relevant subset

                selected_features = self.scaler.transform(song_row[self.features].to_frame().T)
                distances, indices = knn_model_genre_or_subgenre.kneighbors(selected_features)
                closest_songs = relevant_df.iloc[indices[0][1:]]

                new_recommendations = closest_songs[
                    (~closest_songs.index.isin(recommended_indices)) &
                    (~closest_songs[['track_name', 'track_artist']].apply(tuple, axis=1).isin(recommendations[['track_name', 'track_artist']].apply(tuple, axis=1).tolist())) &
                    (closest_songs['track_name'].str.lower() != song_name.lower()) &
                    (closest_songs['track_artist'].str.lower() != artist_name.lower())
                ].drop_duplicates(subset=['track_name', 'track_artist'])

                if not new_recommendations.empty:
                    if recommendations.empty:
                        recommendations = new_recommendations.copy()
                    else:
                        recommendations = pd.concat([recommendations, new_recommendations])
                    recommended_indices.update(new_recommendations.index)

                if len(recommendations) >= songs_to_return or n_neighbors >= len(relevant_df):
                    break
                else:
                    n_neighbors = min(len(relevant_df), n_neighbors + 5)

            return recommendations[['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'track_album_release_date']].head(songs_to_return)

# API Endpoint
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()
    print(data)
    spotify_url = data.get("spotify_url")
    if not spotify_url:
        return jsonify({"error": "Spotify URL is required."}), 400
    recommender = KNNRecommender()
    track_id = recommender.extract_track_id(spotify_url)
    if not track_id:
        return jsonify({"error": "Invalid Spotify URL."}), 400

    recommendations = recommender.recommend_songs(track_id, recommender.df)
    artist_name, track_name, genres, release_year = recommender.get_spotify_song_info(track_id)
    song_info = recommender.get_spotify_info(track_name, artist_name)

    if recommendations.empty or recommendations is None:
        return jsonify({"error": "No recommendations found."}), 404
    
    print(recommendations)

    recommendations_list = []
    for _, row in recommendations.iterrows():
        # Get Spotify info for each recommendation
        spotify_info = recommender.get_spotify_info(row['track_name'], row['track_artist'])
        if spotify_info:
            recommendations_list.append({
                "title": spotify_info['title'],
                "artist": spotify_info['artist'],
                "url": spotify_info['url'],
                "imageUrl": spotify_info['imageUrl'],
                "genre": spotify_info['genre'],
                "release_date": spotify_info['release_date']
            })
        if not recommendations_list:
            return None
    
    return jsonify({
        "input_track_title": song_info['title'],
        "input_track_artist": song_info['artist'],
        "input_track_url": song_info['url'],
        "input_track_imageUrl": song_info['imageUrl'],
        "input_track_genre": song_info['genre'],
        "input_track_release_date": song_info['release_date'],
        "recommendations": recommendations_list
    })

@app.route('/health')
def health_check():
    return 'OK', 200

app.run(host='0.0.0.0', port=5000, debug=True)
