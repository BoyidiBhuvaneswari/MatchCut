from flask import Flask, request, jsonify, render_template
import sqlite3
import numpy as np
import tensorflow as tf
from textblob import TextBlob
from hybrid_movie_recommendation_engine import build_ncf, calculate_hybrid_score
import threading
from omdb_client import omdb_search_multiple, omdb_get_by_imdb_id, cache_movie_in_db
app = Flask(__name__)

USER_SLOTS = 500
MOVIE_SLOTS = 500

# Keras/TensorFlow models are not guaranteed thread-safe. Training now
# happens in a background thread (see api_rate and the startup block
# below) so it doesn't block HTTP responses, but that means a background
# fit() call and a request-thread predict() call could hit the model at
# the same instant. Without this lock, that collision can hang BOTH
# calls indefinitely rather than just running them one at a time.
model_lock = threading.Lock()

def get_db_connection():
    conn = sqlite3.connect('movies.db')
    conn.row_factory = sqlite3.Row
    return conn

def user_slot(user_id):
    return user_id % USER_SLOTS

def movie_slot(movie_id):
    return movie_id % MOVIE_SLOTS

def train_model_from_db():
    """(Re)trains the NCF model on every rating currently in the DB.
    Our dataset is tiny (dozens of rows at most for a demo), so a full
    retrain on every new rating is cheap and gives the effect of the
    model 'learning' live. A production system with real traffic would
    instead retrain on a schedule (e.g. nightly batch job), not per-request.

    IMPORTANT: this is called from a background thread (see api_rate and
    the startup block below), never directly in a request/startup path —
    on Render's free-tier CPU, running this synchronously was blocking
    long enough to trip gunicorn's worker timeout, causing a boot crash loop."""
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM ratings').fetchall()
    conn.close()
    if not rows:
        return
    X = np.array([[user_slot(r['user_id']), movie_slot(r['movie_id'])] for r in rows])
    y = np.array([r['rating'] for r in rows])
    with model_lock:
        ncf_model.fit([X[:, 0], X[:, 1]], y, epochs=5, batch_size=1, verbose=0)
    print(f"Model retrained on {len(rows)} ratings")


def ensure_db_ready():
    """On a fresh deploy, movies.db won't exist yet (it's gitignored).
    Create the tables if they're missing, so the app can start cleanly."""
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS movies (id INTEGER PRIMARY KEY, title TEXT, genre TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS ratings (user_id INTEGER, movie_id INTEGER, rating REAL)')
    conn.commit()
    conn.close()

ensure_db_ready()
ncf_model = build_ncf(USER_SLOTS, MOVIE_SLOTS)
# Run the initial training in a background thread instead of blocking
# module import (which happens during gunicorn worker boot). Doing this
# synchronously was pushing worker startup past gunicorn's timeout on
# Render's free tier, causing repeated WORKER TIMEOUT / SIGKILL crashes
# before the app ever got a chance to serve a single request.
threading.Thread(target=train_model_from_db, daemon=True).start()

def get_sentiment_score(text):
    analysis = TextBlob(text)
    return (analysis.sentiment.polarity + 1) / 2

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/movies', methods=['GET'])
def api_movies():
    conn = get_db_connection()
    rows = conn.execute('SELECT id, title FROM movies ORDER BY title').fetchall()
    conn.close()
    return jsonify([{"id": r["id"], "title": r["title"]} for r in rows])

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400

    conn = get_db_connection()
    existing = conn.execute(
        'SELECT id FROM users WHERE LOWER(name) = LOWER(?)', (name,)
    ).fetchone()

    if existing:
        user_id = existing['id']
    else:
        cur = conn.execute('INSERT INTO users (name) VALUES (?)', (name,))
        user_id = cur.lastrowid
        conn.commit()

    conn.close()
    return jsonify({"user_id": user_id, "name": name})

@app.route('/api/rated_movies', methods=['GET'])
def api_rated_movies():
    """Which movies this user has already rated, so the frontend can
    grey them out or exclude them from the 'rate a movie' list."""
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400
    conn = get_db_connection()
    rows = conn.execute('SELECT DISTINCT movie_id FROM ratings WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    return jsonify([r['movie_id'] for r in rows])

@app.route('/api/search', methods=['GET'])
def api_search():
    query = (request.args.get('q') or '').strip()
    if not query:
        return jsonify({"results": []})

    conn = get_db_connection()
    local_matches = conn.execute(
        'SELECT id, title FROM movies WHERE title LIKE ? ORDER BY title LIMIT 20',
        (f'%{query}%',)
    ).fetchall()
    conn.close()

    if local_matches:
        return jsonify({
            "results": [{"id": m["id"], "title": m["title"]} for m in local_matches],
            "source": "local"
        })

    # Nothing local — search OMDb (fuzzy/partial match), cache what we find
    omdb_hits = omdb_search_multiple(query)
    if not omdb_hits:
        return jsonify({"results": [], "source": "none"})

    results = []
    for hit in omdb_hits[:8]:  # cap to avoid burning quota on one search
        details = omdb_get_by_imdb_id(hit["imdbID"])
        if details:
            movie_id = cache_movie_in_db(details)
            results.append({"id": movie_id, "title": details["Title"]})

    return jsonify({"results": results, "source": "omdb"})

@app.route('/api/rate', methods=['POST'])
def api_rate():
    """Step 1 of a real recommender: log a rating for a movie the user
    has ALREADY watched. This is training data, not a recommendation —
    no score is shown back to the user, just a confirmation."""
    data = request.get_json() or {}
    try:
        user_id = int(data.get('user_id'))
        movie_id = int(data.get('movie_id'))
        stars = int(data.get('stars'))
    except (TypeError, ValueError):
        return jsonify({"error": "user_id, movie_id, and stars must be numbers"}), 400

    if not (1 <= stars <= 5):
        return jsonify({"error": "stars must be between 1 and 5"}), 400

    review_text = data.get('review', '')

    conn = get_db_connection()
    movie_row = conn.execute('SELECT title FROM movies WHERE id = ?', (movie_id,)).fetchone()
    user_row = conn.execute('SELECT id FROM users WHERE id = ?', (user_id,)).fetchone()
    if not movie_row:
        conn.close()
        return jsonify({"error": "Unknown movie_id"}), 400
    if not user_row:
        conn.close()
        return jsonify({"error": "Unknown user_id"}), 400

    rating_value = stars / 5.0
    conn.execute('INSERT INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)',
                 (user_id, movie_id, rating_value))
    conn.commit()
    conn.close()

    # Retrain in the background so the response returns immediately —
    # running this in-request was blocking long enough on Render's free
    # tier to hang the browser / trip the worker timeout.
    threading.Thread(target=train_model_from_db, daemon=True).start()

    return jsonify({"message": f"Saved your {stars}-star rating for {movie_row['title']}!"})

def get_user_genre_preferences(user_id, conn):
    """Look at movies this user rated highly (>=0.6, i.e. 3+ stars) and
    tally which genres show up most. Returns {genre: avg_rating}."""
    rows = conn.execute('''
        SELECT m.genre, r.rating
        FROM ratings r
        JOIN movies m ON r.movie_id = m.id
        WHERE r.user_id = ?
    ''', (user_id,)).fetchall()

    genre_totals = {}
    genre_counts = {}
    for row in rows:
        genre = row['genre'] or 'Unknown'
        genre_totals[genre] = genre_totals.get(genre, 0) + row['rating']
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    return {g: genre_totals[g] / genre_counts[g] for g in genre_totals}

@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    """Step 2 of a real recommender: predict scores for movies the user
    HASN'T rated yet, ranked highest first. This is the actual useful
    output — 'here's what you'd probably like next'."""
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    conn = get_db_connection()
    user_row = conn.execute('SELECT id FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user_row:
        conn.close()
        return jsonify({"error": "Unknown user_id"}), 400

    all_movies = conn.execute('SELECT id, title FROM movies').fetchall()
    rated_ids = {r['movie_id'] for r in
                 conn.execute('SELECT movie_id FROM ratings WHERE user_id = ?', (user_id,)).fetchall()}
    conn.close()

    unrated = [m for m in all_movies if m['id'] not in rated_ids]
    if not unrated:
        return jsonify({"recommendations": [], "message": "You've rated everything we have — nothing left to recommend!"})

    conn2 = get_db_connection()
    genre_prefs = get_user_genre_preferences(user_id, conn2)
    movie_genres = {row['id']: row['genre'] for row in conn2.execute('SELECT id, genre FROM movies').fetchall()}
    conn2.close()

    # Predict for ALL unrated movies in a single batched call instead of
    # looping model.predict() once per movie. Each individual predict()
    # call carries fixed TensorFlow overhead that doesn't shrink for a
    # batch of 1 — looping it per movie was slow enough on Render's
    # limited free-tier CPU to hang the request indefinitely.
    user_slots_batch = np.array([user_slot(user_id)] * len(unrated))
    movie_slots_batch = np.array([movie_slot(m['id']) for m in unrated])
    with model_lock:
        ncf_scores = ncf_model.predict([user_slots_batch, movie_slots_batch], verbose=0).flatten()

    results = []
    for m, ncf_score in zip(unrated, ncf_scores):
        genre = movie_genres.get(m['id'], 'Unknown')
        genre_score = genre_prefs.get(genre, 0.5)  # neutral 0.5 if genre unseen

        # Hybrid: mostly collaborative, nudged by genre affinity
        final_score = (0.75 * float(ncf_score)) + (0.25 * genre_score)
        results.append({"movie": m['title'], "genre": movie_genres.get(m['id'], ''), "match_percent": round(final_score * 100)})

    results.sort(key=lambda r: r['match_percent'], reverse=True)
    return jsonify({"recommendations": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)