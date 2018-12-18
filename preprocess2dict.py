import pandas as pd
import pickle
from sklearn.utils import shuffle


df = pd.read_csv('ml-latest-small/ratings_preprocessed.csv')

print(df.head())

#set max  number of users and max number of movies
n_users = max(df.userId) + 1
n_movies = max(df.movieId) + 1

#split train and test sets
df = shuffle(df)
cutoff = int(len(df) * 0.8)
train = df.iloc[:cutoff]
test = df.iloc[cutoff:]

#a dictionary that tells which users have rated which movies
user2movie = {}

#a dictionary which tells us which movies where rated by which user
movie2user = {}

#a dictionary which gives us rating for specific movie by specific user
usermovie2rating = {}


def update_user2movie_and_movie2user(row):

    user = int(row.userId)
    movie = int(row.movieId)

    if user not in user2movie:
        user2movie[user] = [movie]
    else:
        user2movie[user].append(movie)

    if movie not in movie2user:
        movie2user[movie] = [user]
    else:
        movie2user[movie].append(user)

    usermovie2rating[(user, movie)] = row.rating

train.apply(update_user2movie_and_movie2user, axis = 1)

# test ratings dictionary
usermovie2rating_test = {}

def update_usermovie2rating_test(row):

    user = int(row.userId)
    movie = int(row.movieId)

    usermovie2rating_test[(user, movie)] = row.rating

test.apply(update_usermovie2rating_test, axis = 1)


with open('user2movie.json', 'wb') as f:
    pickle.dump(user2movie, f)

with open('movie2user.json', 'wb') as f:
    pickle.dump(movie2user, f)

with open('usermovie2rating.json', 'wb') as f:
    pickle.dump(usermovie2rating, f)

with open('usermovie2rating_test.json', 'wb') as f:
    pickle.dump(usermovie2rating_test, f)