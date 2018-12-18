import pandas as pd

#import
df = pd.read_csv('ml-latest-small/ratings.csv')

#make userids start from 0
df['userId'] = df['userId'] - 1

#convert movieids to unique values starting from 0
unique_movie_ids = set(df.movieId.values)
unique_movie_ids = list(unique_movie_ids)
movie_idx_dict = {}
count = 0

for i in unique_movie_ids:
    movie_idx_dict[i] = count
    count += 1

df['movie_idx'] = df.apply(lambda row: movie_idx_dict[row.movieId], axis= 1)

df = df.drop(columns='timestamp')

df.to_csv('ml-latest-small/ratings_preprocessed.csv', index = False)

