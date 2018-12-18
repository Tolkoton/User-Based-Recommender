import pickle
import pandas as pd
import numpy as np
from sortedcontainers import SortedList


#load data
with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)
with open('movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)
with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)
with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)


#find number of users and movies
df = pd.read_csv('ml-latest-small/ratings_preprocessed.csv')
N_users = max(df.userId) + 1
M_movies = max(df.movieId) + 1

#init variables
K_neighboors = 25
common_movies_limit = 5
neighboors = []
averages = []
deviations = []

for i in range(N_users):
    #find 25 closest movies
    movies_i = user2movie[i]
    movies_i_set = set(movies_i)

    #find deviations for particular movies particular user rated. sigma_i - first part of Pearsons corr coef denominator
    ratings_i = {movie: usermovie2rating[(i, movie)] for movie in movies_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {movie: (rating - avg_i) for movie, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    averages.append(avg_i)
    deviations.append(dev_i)

    sorted_list = SortedList()

    for j in range(N_users):
        movies_j = user2movie[j]
        movies_j_set = set(movies_j)
        common_movies = (movies_i_set & movies_j_set)

        if len(common_movies) > common_movies_limit:
            ratings_j = {movie:usermovie2rating[(j, movie)] for movie in movies_j}
            avg_j = np.mean(list(ratings_j.values()))
            dev_j = {movie:(rating - avg_j) for movie, rating in ratings_j.items()}
            dev_j_values = np.array(list(dev_j.values()))
            sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

            numerator = sum(dev_i[m] * dev_j[m] for m in common_movies)
            denominator = (sigma_i * sigma_j)
            wij = numerator / denominator #Pearsons correlation coefficient for users i and j

            # add ith users correletaion with j's user to list
            # insert into sorted_list. negate wij since list is sorted ascending.
            sorted_list.add((-wij, j))
            if len(sorted_list) > K_neighboors:
                del sorted_list[-1]


        #save neighboors
        neighboors.append(sorted_list)



#using neighboors calvulate train and test MSE
def predict(user_i, movie_m):

    numerator = 0
    denominator = 0

    for neg_weight, j in neighboors[user_i]:
        #weight in neighboors is stored in negative
        try:
            numerator += -neg_weight * deviations[j][movie_m]
            denominator += abs(neg_weight)
        except KeyError:
            pass

    #if no common rated movies, prediction is just an average of users ratings
    if denominator == 0:
        prediction = averages[user_i]
    else:
        prediction = numerator / denominator + averages[user_i]

    prediction = min(5, prediction)
    prediction = max(0.5, prediction)

    return prediction

#evaluate predictions for train set
train_predictions = []
train_y = []

for (user_i, movie_m), y in usermovie2rating.items():

    prediction = predict(user_i, movie_m)
    train_predictions.append(prediction)
    train_y.append(y)


#evaluate predictions for test set
test_predictions = []
test_y = []

for (user_i, movie_m), i in usermovie2rating.items():
    prediction = predict(user_i, movie_m)
    test_predictions.append(prediction)
    test_y.append(y)

#calculate accupacy
def mse(predictions, y):
    predictions = np.array(predictions)
    y = np.array(y)
    mse = np.mean((predictions - y)**2)
    return mse

print('train mse:', mse(train_predictions, train_y))
print('test mse:', mse(test_predictions, test_y))

