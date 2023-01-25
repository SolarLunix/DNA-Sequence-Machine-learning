import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV as CCCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# - - - - - - - FINALS - - - - - - - 
TEST_SIZE = 0.10


# - - - - - - - METHODS - - - - - - - 
def read_data(filename: str) -> list:
    data = pd.read_table(filename)
    split_data = []
    for i in range(7):
        split_data.append(data.loc[data["class"] == i, "sequence"].to_list())
    return split_data

def Kmers_funct(seq, size=6):
    data = [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]
    return ' '.join(data)

def split_combine_data(x, y, x_train, x_test, y_train, y_test):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = TEST_SIZE, random_state=42)
    x_train.extend(X_train)
    x_test.extend(X_test)
    y_train.extend(Y_train)
    y_test.extend(Y_test)
    return x_train, x_test, y_train, y_test

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted', zero_division=True)
    recall = recall_score(y_test, y_predicted, average='weighted', zero_division=True)
    f1 = f1_score(y_test, y_predicted, average='weighted', zero_division=True)
    return accuracy, precision, recall, f1

# - - - - - - - LOGIC - - - - - - - 
human = read_data('/mnt/ssd/Repos/DNA-Sequence-Machine-learning/human_data.txt')
chimp = read_data('/mnt/ssd/Repos/DNA-Sequence-Machine-learning/chimp_data.txt')
dog = read_data('/mnt/ssd/Repos/DNA-Sequence-Machine-learning/dog_data.txt')

classifiers = []
t_human = []
t_chimp = []
t_dog = []

for i, (h, c, d) in enumerate(zip(human, chimp, dog)):
    h = [Kmers_funct(h[x], 6) for x in range(60)]
    c = [Kmers_funct(c[x], 6) for x in range(60)]
    d = [Kmers_funct(d[x], 6) for x in range(60)]

    t_human.append(h)
    t_chimp.append(c)
    t_dog.append(d)


print("Data Shapes, human, chimp, dog:", np.array(t_human).shape, np.array(t_chimp).shape, np.array(t_dog).shape)

train_outputs = np.zeros((162, 21))
train_lables = np.zeros((162))
test_outputs = np.zeros((18, 21))
test_lables = np.zeros((18))

for class_num, (h, c, d) in enumerate(zip(t_human, t_chimp, t_dog)):
    print("\n------- ------- ------- ------- ------- ------- -------\n")
    h_labels = [0 for _ in h]
    c_labels = [1 for _ in c]
    d_labels = [2 for _ in d]

    print(f"Class: {class_num} \n")
    print(f"\tHuman \t{len(h)} \t{int(len(h)-(len(h)*TEST_SIZE))} \t{int(len(h)*TEST_SIZE)}")
    print(f"\tChimp \t{len(c)} \t{int(len(c)-(len(c)*TEST_SIZE))} \t{int(len(c)*TEST_SIZE)}")
    print(f"\tDog \t{len(d)} \t{int(len(d)-(len(d)*TEST_SIZE))} \t{int(len(d)*TEST_SIZE)}")

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    x_train, x_test, y_train, y_test = split_combine_data(h, h_labels, x_train, x_test, y_train, y_test)
    x_train, x_test, y_train, y_test = split_combine_data(c, c_labels, x_train, x_test, y_train, y_test)
    x_train, x_test, y_train, y_test = split_combine_data(d, d_labels, x_train, x_test, y_train, y_test)

    cv = CountVectorizer(ngram_range=(6,6))
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    print(f"\t---------------------------- \n\tTraining {x_train.shape} \n\tTesting {x_test.shape}")
    classifier = CCCV(LinearSVC(loss='squared_hinge', C=0.5, max_iter=100000))
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print(f"\t----------------------------")
    print(f"\tAccuracy \t{accuracy:.3f} \n\tPrecision \t{precision:.3f} \n\tRecall \t\t{recall:.3f} \n\tF1 \t\t{f1:.3f}")
    print("\nConfusion Matrix \n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

    for i, (x, y) in enumerate(zip(classifier.predict_proba(x_train), y_train)):
        s = class_num * 3
        train_outputs[i][s:s+3] += x
        train_lables[i] = y
    
    for i, (x, y) in enumerate(zip(classifier.predict_proba(x_test), y_test)):
        s = class_num * 3
        test_outputs[i][s:s+3] += x
        test_lables[i] = y

print("\n------- ------- ------- ------- ------- ------- -------\n")
print("Final Predictions\n")
print(f"\tTraining \t{train_outputs.shape} \t{train_lables.shape}")
print(f"\tTestin6 \t{test_outputs.shape} \t{test_lables.shape}")
print(f"\t----------------------------")
classifier = CCCV(LinearSVC(loss='squared_hinge', C=0.5, max_iter=100000))
classifier.fit(train_outputs, train_lables)

y_pred = classifier.predict(test_outputs)
accuracy, precision, recall, f1 = get_metrics(test_lables, y_pred)
print(f"\tAccuracy \t{accuracy:.3f} \n\tPrecision \t{precision:.3f} \n\tRecall \t\t{recall:.3f} \n\tF1 \t\t{f1:.3f}")
print("\nConfusion Matrix \n")
print(pd.crosstab(pd.Series(test_lables, name='Actual'), pd.Series(y_pred, name='Predicted')))
print("\n------- ------- ------- ------- ------- ------- -------\n")

y_pred = classifier.predict(train_outputs)
accuracy, precision, recall, f1 = get_metrics(train_lables, y_pred)
print(f"\tAccuracy \t{accuracy:.3f} \n\tPrecision \t{precision:.3f} \n\tRecall \t\t{recall:.3f} \n\tF1 \t\t{f1:.3f}")
print("\nConfusion Matrix \n")
print(pd.crosstab(pd.Series(train_lables, name='Actual'), pd.Series(y_pred, name='Predicted')))
print("\n------- ------- ------- ------- ------- ------- -------\n")