import os
import requests
import sqlite3

OMDB_API_KEY = os.environ.get("OMDB_API_KEY")
OMDB_URL = "http://www.omdbapi.com/"


def omdb_search_multiple(query):
    """Search OMDb for titles matching a (possibly partial) query.
    Uses the 's=' search endpoint, which does fuzzy/partial matching,
    unlike 't=' which needs a near-exact title. Returns a list of
    dicts with at least Title/imdbID, or an empty list."""
    if not OMDB_API_KEY:
        print("OMDb search skipped: OMDB_API_KEY is not set")
        return []
    try:
        resp = requests.get(OMDB_URL, params={
            "apikey": OMDB_API_KEY,
            "s": query,
            "type": "movie"
        }, timeout=5)
        data = resp.json()
        if data.get("Response") == "True":
            return data.get("Search", [])
        print(f"OMDb search error for '{query}': {data.get('Error')}")
        return []
    except requests.RequestException as e:
        print(f"OMDb request failed for '{query}': {e}")
        return []


def omdb_get_by_imdb_id(imdb_id):
    """Fetch full details for one title by its IMDb ID (from a search
    result). This gives us Genre etc. that the search endpoint omits."""
    if not OMDB_API_KEY:
        return None
    try:
        resp = requests.get(OMDB_URL, params={
            "apikey": OMDB_API_KEY,
            "i": imdb_id
        }, timeout=5)
        data = resp.json()
        if data.get("Response") == "True":
            return data
        return None
    except requests.RequestException as e:
        print(f"OMDb detail lookup failed for '{imdb_id}': {e}")
        return None


def omdb_search_by_title(title):
    """Look up a single title on OMDb using the exact-title endpoint.
    Kept for the seed script, where titles are already exact.
    Returns a dict or None."""
    if not OMDB_API_KEY:
        return None
    try:
        resp = requests.get(OMDB_URL, params={
            "apikey": OMDB_API_KEY,
            "t": title,
            "type": "movie"
        }, timeout=5)
        data = resp.json()
        if data.get("Response") == "True":
            return data
        print(f"OMDb error for '{title}': {data.get('Error')}")
        return None
    except requests.RequestException as e:
        print(f"Request failed for '{title}': {e}")
        return None


def cache_movie_in_db(omdb_data):
    """Insert an OMDb result into our local movies table so future
    searches for it are instant and don't cost an API call."""
    conn = sqlite3.connect('movies.db')
    cur = conn.cursor()
    existing = cur.execute(
        'SELECT id FROM movies WHERE LOWER(title) = LOWER(?)', (omdb_data["Title"],)
    ).fetchone()
    if existing:
        conn.close()
        return existing[0]

    genre = (omdb_data.get("Genre") or "Unknown").split(",")[0].strip()
    cur.execute('INSERT INTO movies (title, genre) VALUES (?, ?)',
                (omdb_data["Title"], genre))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id