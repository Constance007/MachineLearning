import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import random

# Celisus
#x = list(range(0,10))
x = list(range(0,50)) # Estimate temp


# Fahrenheit
# y = [1.8*F + 32 for F in x] # Initial
y = [1.8*F + 32 + random.randint(-3, 3) for F in x] # Introduce noise
#print(f'X: {x}')
#print(f'Y: {y}')

plt.plot(x, y, '-*r')
#plt.show() C/O to  estimate temperature

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
#print(x)

xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size=0.2)
#print(xTrain.shape)

# Define model
model = linear_model.LinearRegression()

# Fit model
model.fit(xTrain, yTrain)
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# Accuracy
accuracy = model.score(xVal, yVal)
print(f'Accuracy: {round(accuracy * 100,2)}')

# Estimate temperature
x = x.reshape(1,-1)[0]
m = model.coef_[0][0]
c = model.intercept_[0]
y = [m*F + c for F in x]

plt.plot(x, y, '-*b')
plt.show()

