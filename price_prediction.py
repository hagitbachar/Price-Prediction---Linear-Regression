import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("kc_house_data.csv")

#drop all lines that has one null or more
data.dropna(how = 'any', inplace = True)

#drop the "id" column, because the id didn't affect the house price
data.drop(['id'],1 , inplace = True)

#drop the "view" column, because the view didn't affect the house price
data.drop(['view'],1 , inplace = True)

#drop all the houses that have a non positive price
data.drop(data[data['price'] <= 0].index, inplace = True)

#drop all the houses that have a non positive bedrooms
data.drop(data[data['bedrooms'] <= 0].index, inplace = True)

#drop all the houses that have a non positive bathrooms
data.drop(data[data['bathrooms'] <= 0].index, inplace = True)

#drop the "date" columns, because the date didn't affect the house price
data.drop(['date'], 1, inplace = True)

#convert the 'zipcode' column to a binary coulmns
data = pd.get_dummies(data, columns = ['zipcode'])

#append a row of ones
ones= []
for i in range(len(data)):
    ones.append(1)

data.insert(0, 'ones', ones)

# print(data.describe())



squared_loss_training = []
squared_loss_test = []

for i in range(1, 100):
    train = 0
    test = 0
    for j in range(10):
        training_data = data.sample(frac = i/100)
        test_data = data.drop(training_data.index)
        training_prices = training_data["price"]
        training_data.drop(["price"], 1,inplace = True)
        training_data_matrix = training_data.as_matrix()
        training_prices_vector = training_prices.as_matrix()
        w = np.dot(np.linalg.pinv(training_data_matrix), training_prices_vector)
        train += ((((np.dot(training_data_matrix, w) - training_prices_vector)**2).mean())**0.5)

        test_prices = test_data["price"]
        test_data.drop(["price"], 1,inplace = True)
        test_data_matrix = test_data.as_matrix()
        test_prices_vector = test_prices.as_matrix()
        test += ((((np.dot(test_data_matrix, w) - test_prices_vector)**2).mean())**0.5)

    squared_loss_training.append(train / 10)
    squared_loss_test.append(test / 10)


plt.plot(range(1, 100), squared_loss_training, label = "training error")
plt.plot(range(5, 100), squared_loss_test[4:], label = "test error")
plt.legend()
plt.xlabel("The percentage of the data")
plt.ylabel("error value")
plt.show()
