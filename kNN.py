#Kian Chryslyr Q. Cadungog
#2BSCS-A

def euclidean_distance(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

def k_nearest_neighbors(train_data, train_labels, test_point, k):
    distances = [(euclidean_distance(test_point, train_data[i]), train_labels[i]) for i in range(len(train_data))]
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    return max(set(k_nearest), key=k_nearest.count)

train_data = [
    [1, 2], [2, 3], [3, 4], [5, 6], [8, 8], [9, 10],
    [10, 12], [15, 16], [16, 18], [17, 19]
]
train_labels = ['Ginger Tea', 'Ginger Tea', 
                'Ginger Tea', 'Chocolate', 
                'Chocolate', 'Chocolate', 
                'Chocolate', 'Water Refilling', 
                'Water Refilling', 'Water Refilling']
test_point = [10, 10]
k = 5
predicted_label = k_nearest_neighbors(train_data, train_labels, test_point, k)
print(f"Predicted label for {test_point}: {predicted_label}")
