import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]  
y = [2, 4, 5, 4, 5] 

# Step 1
x_mean = sum(x) / len(x)
y_mean = sum(y) / len(y)

# Step 2
numerator = 0
denominator = 0

for i in range(len(x)):
    numerator += (x[i] - x_mean) * (y[i] - y_mean)
    denominator += (x[i] - x_mean) ** 2

m = numerator / denominator
b = y_mean - (m * x_mean)

# Step 3
y_hat = [] 
for xi in x:
    y_hat.append(m * xi + b)

print("Slope (m):", m)
print("Intercept (b):", b)
print(f"Regression Equation: y = {m:.2f}x + {b:.2f}")

# Step 4
plt.scatter(x, y, color="blue", label="Data Points")
plt.plot(x, y_hat, color="red", label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()