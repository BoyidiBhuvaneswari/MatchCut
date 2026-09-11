import time
from omdb_client import omdb_search_by_title, cache_movie_in_db

SEED_TITLES = [
    "Inception", "The Dark Knight", "Interstellar", "Parasite",
    "RRR", "Baahubali: The Beginning", "3 Idiots", "Dangal",
    "Spirited Away", "Pan's Labyrinth", "The Godfather", "Whiplash",
    # add more across languages/industries as you like
]

if __name__ == '__main__':
    for title in SEED_TITLES:
        data = omdb_search_by_title(title)
        if data:
            cache_movie_in_db(data)
            print(f"Cached: {data['Title']}")
        else:
            print(f"Not found: {title}")
        time.sleep(0.3)  # be polite to the free API