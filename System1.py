import csv
import numpy as np

# reads the train dataset csv file, taking it in as a numpy array
data = np.genfromtxt('train_100k_withratings_new.csv', delimiter=',')

# then shuffles this data in case the order could negatively impact the results
# keeping the seed constant to keep the output MAE consistent
np.random.seed(42)
np.random.shuffle(data)

# reads the test dataset csv file, taking it in as a numpy array
test_data = np.genfromtxt('test_100k_withoutratings_new.csv', delimiter=',')

# creates and opens a new results csv to save the outcome of the test set
results = open('results.csv', 'w', newline='')

# creates the csv writer
writer = csv.writer(results)

# creates the user-item matrix
n_users = int(np.max(data[:, 0]))
n_items = int(np.max(data[:, 1]))
# initialises the matrix, starting off with all values being 0
user_item_matrix = np.zeros((n_users, n_items))
# then sets the appropriate rating values in the correct places
for row in data:
    user_id = int(row[0] - 1)
    item_id = int(row[1] - 1)
    rating = row[2]
    user_item_matrix[user_id][item_id] = rating

# calculates the cosine similarity between each pair of users, using the user-item matrix
# and stores the new matrix in user_similarity, of which also starts off as an empty matrix of 0s
user_similarity = np.zeros((n_users, n_users))
for i in range(n_users):
    for j in range(i, n_users):
        if i == j:
            user_similarity[i][j] = 1
        else:
            dot_product = np.dot(user_item_matrix[i], user_item_matrix[j])
            norm_i = np.linalg.norm(user_item_matrix[i])
            norm_j = np.linalg.norm(user_item_matrix[j])
            cosine_similarity = dot_product / (norm_i * norm_j)
            user_similarity[i][j] = cosine_similarity
            user_similarity[j][i] = cosine_similarity

# calculates the cosine similarity between each pair of items
# very similar to the previous block
# however vectors used for the similarity calculation are now the columns of the user-item matrix for each item
item_similarity = np.zeros((n_items, n_items))
for i in range(n_items):
    for j in range(i, n_items):
        if i == j:
            item_similarity[i][j] = 1
        else:
            dot_product = np.dot(user_item_matrix[:,i], user_item_matrix[:,j])
            norm_i = np.linalg.norm(user_item_matrix[:,i])
            norm_j = np.linalg.norm(user_item_matrix[:,j])
            cosine_similarity = dot_product / (norm_i * norm_j)
            item_similarity[i][j] = cosine_similarity
            item_similarity[j][i] = cosine_similarity

# predict ratings for test set using weighted average of similar users ratings
predictions = []
for row in test_data:
    # 1 is added so the id can be used for indexing; to start from 0
    user_id = int(row[0] - 1)
    item_id = int(row[1] - 1)
    timestamp = row[2]
    k = 40
    # find k number of most similar users, excluding itself
    similar_users = np.argsort(user_similarity[user_id])[::-1][1:k+1]
    weighted_ratings = []
    similarities = []
    for su in similar_users:
        # if the similar user has rated the current item, calculate their weighted rating and similarity to current user
        if user_item_matrix[su][item_id] != 0:
            weighted_ratings.append(user_item_matrix[su][item_id] * user_similarity[user_id][su])
            similarities.append(user_similarity[user_id][su])

    # if there have been ratings found, cap the number of item ratings to use for the prediction
    if len(weighted_ratings) != 0:
        # find 2k number of most similar items, excluding itself
        similar_items = np.argsort(item_similarity[item_id])[::-1][1:2*k+1]
    else:
        # find all similar items, excluding itself
        similar_items = np.argsort(item_similarity[item_id])[::-1][1:]
    for si in similar_items:
        # if the user has rated that item, it calculates the weighted rating and similarity for the item
        if user_item_matrix[user_id][si] != 0:
            weighted_ratings.append(user_item_matrix[user_id][si] * item_similarity[item_id][si])
            similarities.append(item_similarity[item_id][si])
    # then calculates the prediction based on the weighted ratings and similarities previously retrieved
    if len(weighted_ratings) > 0:
        weighted_ratings = np.array(weighted_ratings)
        similarities = np.array(similarities)
        # calculates the prediction
        prediction = np.sum(weighted_ratings) / np.sum(similarities)
        # makes sure to keep the values between 0.5 and 5
        prediction = max(0.5, min(5, prediction))
        # adds the prediction to the list of predictions
        predictions.append((user_id, item_id, prediction, timestamp))

# to check whether there have been predictions for every entry in the test set
print(len(test_data))
print(len(predictions))

# write the output of the predictions to the results csv file
for p in predictions:
    writer.writerow([p[0] + 1,p[1] + 1,p[2],p[3]])

# finally closes that file
results.close()