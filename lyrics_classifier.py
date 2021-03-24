import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

FILE_PATH = "/Users/suyashramteke/PycharmProjects/ML_MIR/lyrics_3genres.csv"
dataset = pd.read_csv(FILE_PATH)



rap_rows = dataset[dataset['labels'] == 'rap']
rock_rows = dataset[dataset['labels'] == 'rock']
country_rows = dataset[dataset['labels'] == 'country']
# Binary vectors
genre_vec = np.array(dataset.iloc[:, 1:-1])
rap_vec = np.array(rap_rows.iloc[:, 1:-1])
rock_vec = np.array(rock_rows.iloc[:, 1:-1])
country_vec = np.array(country_rows.iloc[:, 1:-1])

# Probability Vectors

rap_rows_sum = (rap_rows.iloc[:, 1:-1]).sum()
rap_prob = rap_rows_sum.astype(float)/1000
rock_rows_sum = (rock_rows.iloc[:, 1:-1]).sum()
rock_prob = rock_rows_sum.astype(float)/1000
country_rows_sum = (country_rows.iloc[:, 1:-1]).sum()
country_prob = country_rows_sum.astype(float)/1000

rap_prob_vec = np.array(rap_prob)
rock_prob_vec = np.array(rock_prob)
country_prob_vec = np.array(country_prob)

rap_words = rap_prob.to_dict()
rock_words = rock_prob.to_dict()
country_words = country_prob.to_dict()
print(rap_words)


"""
print("Conditional Probability vector for rap: \n", rap_prob)
print("Conditional Probability vector for rock: \n", rock_prob)
print("Conditional Probability vector for country :\n", country_prob)
"""

# Calculate the probability of each word in the dictionary given the genre
rap_5 = rap_prob.nlargest(5)
print("Top 5 words for rap :\n", rap_5)
rock_5 = rock_prob.nlargest(5)
print("Top 5 words for rock :\n", rock_5)
country_5 = country_prob.nlargest(5)
print("Top 5 words for country :\n", country_5)

# Naive Bayes Classifier Bernoulli approach
def likelihood(test_song, word_probs_for_genre):
    probability_product = 1.0
    for (i, w) in enumerate(test_song):
        if (w==1):
            probability = word_probs_for_genre[i]
        else:
            probability = 1.0 - word_probs_for_genre[i]
        probability_product *= probability
    return probability_product

def predict(test_song):
    scores = [likelihood(test_song, rap_prob_vec),
             likelihood(test_song, rock_prob_vec),
             likelihood(test_song, country_prob_vec)]
    labels = ['rap', 'rock', 'country']
    return labels[np.argmax(scores)]

# Predict a random track
track_id = np.random.randint(3000)
print("Random track id", track_id)
# test song is a column vector
test_song = genre_vec[track_id]
print("Prediction :", predict(test_song))
print("Label :", dataset["labels"].iloc[track_id])

# Calculate accuracy and confusion matrix

def accuracy(test_set, ground_truth_label, error_label_1, error_label_2):
    score = 0
    error_1 = 0
    error_2 = 0
    for r in test_set:
        if predict(r) == ground_truth_label:
            score += 1
        elif predict(r) == error_label_1:
            error_1 += 1
        else:
            error_2 += 1
    # convert to percentage
    score = score/10.0
    error_1 /= 10.0
    error_2 /= 10.0

    return (score, error_1, error_2)

rap_metric = accuracy(rap_vec, 'rap', 'rock', 'country')
rock_metric = accuracy(rock_vec, 'rock', 'rap', 'country')
country_metric = accuracy(country_vec, 'country', 'rap', 'rock')

print("\nRap Accuracy :", rap_metric[0])
print("Rock Accuracy :", rock_metric[0])
print("Country Accuracy :", country_metric[0])

def confusion_matrix():

    # Important to know how to declare array
    matrix = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            if(i == 0) and (j == 0):
                matrix[i, j] += rap_metric[0]
            elif(i == 1) and (j == 0):
                matrix[i, j] += rock_metric[1]
            elif (i == 2) and (j == 0):
                matrix[i, j] += country_metric[1]
            elif (i == 0) and (j == 1):
                matrix[i, j] += rap_metric[1]
            elif (i == 1) and (j == 1):
                matrix[i, j] += rock_metric[0]
            elif (i == 2) and (j == 1):
                matrix[i, j] += country_metric[2]
            elif (i == 0) and (j == 2):
                matrix[i, j] += rap_metric[2]
            elif (i == 1) and (j == 2):
                matrix[i, j] += rock_metric[2]
            elif (i == 2) and (j == 2):
                matrix[i, j] += country_metric[0]

    return matrix

print("\nConfusion Matrix :\n", confusion_matrix())

# k-fold cross validation, k=10
"""For k fold cross validation, I will need to modify all the functions above and it will be a tedious process.
So I am just describing the process. We would need a loop till 10, in that loop we will have to create test loop as 10 
percent of the original set and the rest 90 percent would be training set. At every iteration, we would modify the
test set to the next batch of 10 percent and so on. Finally we will take the average and compute the probability results"""

# Generate few random words from our Naive Bayes Classifier
def generate():
    print('Random rap', [w for (i, w) in enumerate(rap_words) if np.greater(rap_prob_vec, np.random.rand(30))[i]])
    print('Random rock', [w for (i, w) in enumerate(rock_words) if np.greater(rock_prob_vec, np.random.rand(30))[i]])
    print('Random country', [w for (i, w) in enumerate(country_words) if np.greater(country_prob_vec, np.random.rand(30))[i]])

generate()


