import numpy as np
import csv


# Recommender system using Mini-Batch Stochastic Gradient Descent 
# This is the most effective implementation of SGD when considering the large size of the training set
# - making sure it is time and memory appropriate to run
# It combines the efficiency of SGD but the stability of gradient descent
# It focuses on minimizing the cost function, in this case the error calculated for each training example


class RecommenderSystem:
    def __init__(self, num_users, num_items, latent_d, lr, reg):
        self.num_users = num_users
        self.num_items = num_items
        # the number of latent dimensions - the number of features to be learnt for each user and item
        self.latent_d = latent_d
        # the learning rate - the 'step size' for adjusting the U and V matrices
        self.lr = lr
        # the regularization rate - to avoid overfitting
        self.reg = reg
        # latent factor matrices used to represent relationships between users and items
        # each row of the U matrix represents a user, with the columns being the latent factors
        self.U = np.random.normal(size=(self.num_users, self.latent_d))
        # each row of the V matrix represents an item, with the columns being the latent factors
        self.V = np.random.normal(size=(self.num_items, self.latent_d))
        
    def train(self, data, epochs, batch_size):
        for epoch in range(epochs):
            print(epoch)
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                # extracts the user and item pairings in the given batch
                batch_indices = indices[i:i+batch_size]
                batch_data = data[batch_indices]
                batch_user_ids = batch_data[:, 0].astype(int)
                batch_item_ids = batch_data[:, 1].astype(int)
                batch_ratings = batch_data[:, 2]
                # calculates the error between the actual and predicted ratings
                # uses dot product for each element to calculate the predicted ratings
                # then they are subtracted from the actual ratings, for each pairing element in the batch
                errors = batch_ratings - np.sum(self.U[batch_user_ids] * self.V[batch_item_ids], axis=1)
                dU = np.zeros_like(self.U)
                dV = np.zeros_like(self.V)
                # calculates the gradients of the total error of each user and item pairings within the batch
                for j in range(len(batch_indices)):
                    dU[batch_user_ids[j]] += errors[j] * self.V[batch_item_ids[j]]
                    dV[batch_item_ids[j]] += errors[j] * self.U[batch_user_ids[j]]
                # updates the U and V matrices based on the previously caluclated gradients
                # takes into account the learning rate to manage step size and the regularization rate 
                self.U += self.lr * (dU - self.reg * self.U)
                self.V += self.lr * (dV - self.reg * self.V)


    # function to predict the rating for a given user and item combination using the trained recommender system
    def predict(self, user_id, item_id):
        # the dot product of a user from the U matrix and an item from the V matrix calculates a predicted rating for the given pairing
        rating = np.dot(self.U[int(user_id), :], self.V[int(item_id), :])
        # limits the ratings to between 0.5 and 5
        rating = max(rating, 0.5)
        rating = min(rating, 5)
        return rating


# creates and opens a new results csv to save the outcome of the test set
results = open('cw2_results4.csv', 'w', newline='')

# creates the csv writer
writer = csv.writer(results)

# loads the train and test data
# also shifts the user and item ids to start from 0 to be more convenient for training and testing (is later shifted back)
train_data = np.genfromtxt('train_20m_withratings_new.csv', delimiter=',')
train_data[:, :2] -= 1

test_data = np.genfromtxt('test_20m_withoutratings_new.csv', delimiter=',')
test_data[:, :2] -= 1


np.random.shuffle(train_data)
# train_data = data[:int(0.9 * len(data))]
# test_data = data[int(0.9 * len(data)):]

# calculate the number of users and items
num_users = int(np.max(train_data[:, 0])) + 1
num_items = int(np.max(train_data[:, 1])) + 1

# initialize and train the model
model = RecommenderSystem(num_users, num_items, latent_d=30, lr=0.004, reg=0.01)
model.train(train_data, epochs=70, batch_size=20000)

# writes the predictions on the test set to a csv file
for user_id, item_id, timestamp in test_data:
    prediction = model.predict(user_id,item_id)
    writer.writerow([int(user_id)+1, int(item_id)+1, prediction, int(timestamp)])
    

# mae = 0
# for user_id, item_id, rating, timestamp in test_data:
#     predicted_rating = model.predict(user_id, item_id)
#     mae += abs(predicted_rating - rating)
# mae /= len(test_data)
# print(f"Mean absolute error on test set: {mae:.4f}")