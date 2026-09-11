import sqlite3

def create_db():
    conn = sqlite3.connect('movies.db')
    curr = conn.cursor()
    curr.execute('CREATE TABLE IF NOT EXISTS movies (id INTEGER PRIMARY KEY, title TEXT, genre TEXT)')
    curr.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)')
    curr.execute('CREATE TABLE IF NOT EXISTS ratings (user_id INTEGER, movie_id INTEGER, rating REAL)')

    # Adding some sample data
    movies = [(101, 'Inception', 'Sci-Fi'), (102, 'The Dark Knight', 'Action'), (103, 'Interstellar', 'Sci-Fi')]
    curr.executemany('INSERT OR IGNORE INTO movies VALUES (?,?,?)', movies)

    users = [(1, 'Alice'), (2, 'Bob')]
    curr.executemany('INSERT OR IGNORE INTO users VALUES (?,?)', users)

    ratings = [(1, 101, 0.8), (1, 102, 0.7), (2, 102, 0.9), (2, 103, 0.85)]
    curr.executemany('INSERT OR IGNORE INTO ratings VALUES (?,?,?)', ratings)

    conn.commit()
    conn.close()
    print("Database 'movies.db' created with sample movies, users, and ratings!")

if __name__ == '__main__':
    create_db()