dataset = [
#(Plays Games,   Drinks Energy Drink,     Stays Up Late,        Label)
    ("Yes",             "Yes",               "Yes",            "Gamer"),
    ("Yes",             "No",                 "Yes",           "Gamer"),
    ("Yes",             "Yes",                "No",            "Gamer"),
    ("No",              "Yes",                "Yes",           "Not Gamer"),
    ("No",              "No",                 "No",            "Not Gamer"),
    ("No",              "Yes",                "No",            "Not Gamer"),
    ("Yes",             "No",                 "No",            "Gamer"),
    ("No",              "No",                 "Yes",           "Not Gamer")
]

# Step 1: Count 
feature_counts = {"Gamer": {}, "Not Gamer": {}}
class_counts = {"Gamer": 0, "Not Gamer": 0}
vocab = set()

for games, energy, late, label in dataset:
    class_counts[label] += 1

    for feature in (games, energy, late):
        vocab.add(feature)
        if feature in feature_counts[label]:
            feature_counts[label][feature] += 1
        else:
            feature_counts[label][feature] = 1

# Step 2: Compute prior probabilities
total_samples = sum(class_counts.values())
prior_gamer = class_counts["Gamer"] / total_samples
prior_not_gamer = class_counts["Not Gamer"] / total_samples

# Function to calculate likelihood with Laplace smoothing
def feature_probability(feature, label, alpha=1):
    feature_freq = feature_counts[label].get(feature, 0) + alpha
    total_features = sum(feature_counts[label].values()) + alpha * len(vocab)
    return feature_freq / total_features

# Step 3: Classify a new person
def classify(plays_games, drinks_energy, stays_up_late):
    # Compute posterior probabilities
    gamer_prob = prior_gamer
    not_gamer_prob = prior_not_gamer
    
    for feature in (plays_games, drinks_energy, stays_up_late):
        gamer_prob *= feature_probability(feature, "Gamer")
        not_gamer_prob *= feature_probability(feature, "Not Gamer")
    
    return "Gamer" if gamer_prob > not_gamer_prob else "Not Gamer"

# Testing
new_person = ("Yes", "Yes", "No")  # This person plays games, drinks energy drinks, but doesn't stay up late.
print(f"Person: {new_person} -> Prediction: {classify(*new_person)}")
