# - - - - - - - IMPORTS - - - - - - - 
import itertools
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# - - - - - - - FINALS - - - - - - - 
TEST_SIZE = .1

# - - - - - - - METHODS - - - - - - - 
def read_data(filename: str, species: int, x=[], y =[]) -> list[list, list]:
    data = pd.read_table(filename)
    y_temp = []
    for i in range(7):
        x.extend(data.loc[data["class"] == i, "sequence"].to_list())
        y_temp.extend(data.loc[data["class"] == i, "class"].to_list())
    for i in range(len(y_temp)):
        y.extend([[y_temp[i], species]])
    return x, y

def split_combine_data(x, y, x_train, x_test, y_train, y_test):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = TEST_SIZE, random_state=42)
    x_train.extend(X_train)
    x_test.extend(X_test)
    y_train.extend(Y_train)
    y_test.extend(Y_test)
    return x_train, x_test, y_train, y_test

def print_data_information(x, y):
    x = np.array(x)
    y = np.array(y)

    unique = np.unique(y, return_counts=True)
    print(f"There are {x.shape[0]} samples split as follows:")
    for i, n in enumerate(unique[1]):
        print(f"\tClass {i}: \t{n}")

def create_list(str_len=20):
    lst = ["".join(x) for x in itertools.product('actg', repeat=str_len)]
    return np.array(lst)

def kmer_count_vector(seq_full, c_vals):
    size = len(c_vals[0])
    counts = np.zeros(c_vals.shape[0])
    for x in range(len(seq_full) - size + 1):
        seq = np.array(list(seq_full[x:x+size].lower()))
        unknown_indices = np.where(seq == 'n')[0]
        unknown_val = len(unknown_indices)
        val = 1 / (1 + len(unknown_indices))
        
        if unknown_val > 0:
            permutations = create_list(unknown_val)
            for per in permutations:
                seq[unknown_indices] = per
                idx = np.where(c_vals == ''.join(seq))[0]
                counts[idx] += val
        else:
            idx = np.where(c_vals == ''.join(seq))[0]
            counts[idx] += val
    
    return counts

def Kmers_funct(seq, size=6):
    data = []
    for x in range(len(seq) - size + 1):
        item = seq[x:x+size].lower()
        item = item.replace('u', 't')
        data.append(item)
    return ' '.join(data)

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted', zero_division=True)
    recall = recall_score(y_test, y_predicted, average='weighted', zero_division=True)
    f1 = f1_score(y_test, y_predicted, average='weighted', zero_division=True)
    return accuracy, precision, recall, f1

def pipeline(x_trainp, x_testp, y_train, y_test, ng=(3,3), k=9):
    # Create Kmers
    '''
    cv = CountVectorizer(ngram_range=ng)

    x_train = [Kmers_funct(x, k) for x in x_train]
    x_train = cv.fit_transform(x_train)

    x_test = [Kmers_funct(x, k) for x in x_test]
    x_test = cv.transform(x_test)
    '''

    cv_vals = create_list(k)
    x_train = [kmer_count_vector(x, cv_vals) for x in x_trainp]
    x_test = [kmer_count_vector(x, cv_vals) for x in x_testp]

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    print(f"------------------------------- \n\tTraining {x_train.shape} \n\tTesting {x_test.shape}")
    classifier = LinearSVC(loss='squared_hinge', C=0.5, max_iter=100000)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print(f"\t----------------------------")
    print(f"\tAccuracy \t{accuracy:.3f} \n\tPrecision \t{precision:.3f} \n\tRecall \t\t{recall:.3f} \n\tF1 \t\t\t{f1:.3f}")
    print("\nConfusion Matrix \n")

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    print("-------------------------------")

def species(X, Y):
    print("\nDetecting Species from any sample")
    x = np.array(X)
    y = np.array(Y)[:, 1]
    x_train = []
    x_test = [] 
    y_train = [] 
    y_test = []

    for c in np.unique(y):
        indexes = [i for i, val in enumerate(y) if val == c]
        x_train, x_test, y_train, y_test = split_combine_data(x[indexes], y[indexes], x_train, x_test, y_train, y_test)
    
    print("Training: ", end="")
    print_data_information(x_train, y_train)
    print("Testing: ", end="")
    print_data_information(x_test, y_test)

    pipeline(x_train, x_test, y_train, y_test)

def gene(X, Y, k=6):
    print("\nDetecting Gene from any sample")
    x = np.array(X)
    y = np.array(Y)[:, 0]
    x_train = []
    x_test = [] 
    y_train = [] 
    y_test = []

    for c in np.unique(y):
        indexes = [i for i, val in enumerate(y) if val == c]
        x_train, x_test, y_train, y_test = split_combine_data(x[indexes], y[indexes], x_train, x_test, y_train, y_test)
    
    print("Training: ", end="")
    print_data_information(x_train, y_train)
    print("Testing: ", end="")
    print_data_information(x_test, y_test)

    pipeline(x_train, x_test, y_train, y_test, k=k)

def gene_species(X, Y, k=6):
    print("\nDetecting Species and Gene from sample")
    x = np.array(X)
    y = [(s*7 + g) for (g, s) in Y]
    y = np.array(y)
    x_train = []
    x_test = [] 
    y_train = [] 
    y_test = []

    for c in np.unique(y):
        indexes = [i for i, val in enumerate(y) if val == c]
        x_train, x_test, y_train, y_test = split_combine_data(x[indexes], y[indexes], x_train, x_test, y_train, y_test)
    
    print("Training: ", end="")
    print_data_information(x_train, y_train)
    print("Testing: ", end="")
    print_data_information(x_test, y_test)

    pipeline(x_train, x_test, y_train, y_test, k=k)

def gene_species2(X, Y, k=6):
    print("\nDetecting Species from Gene sample")
    Y = np.array(Y)
    for c in np.unique(Y[:, 0]):
        indexes = [i for i, val in enumerate(Y[:, 0]) if val == c]
        x = np.array(X)[indexes]
        y = np.array(Y)[indexes, 1]
        x_train = []
        x_test = [] 
        y_train = [] 
        y_test = []

        print(f"Class {c}")
        for n in np.unique(y):
            idx = [i for i, val in enumerate(y) if val == n]
            x_train, x_test, y_train, y_test = split_combine_data(x[idx], y[idx], x_train, x_test, y_train, y_test)
    
        print("Training: ", end="")
        print_data_information(x_train, y_train)
        print("Testing: ", end="")
        print_data_information(x_test, y_test)

        pipeline(x_train, x_test, y_train, y_test, k=k)

# - - - - - - - MAIN - - - - - - - 
def main():
    X, Y = read_data('/mnt/ssd/Repos/DNA-Sequence-Machine-learning/human_data.txt', 0)
    X, Y = read_data('/mnt/ssd/Repos/DNA-Sequence-Machine-learning/chimp_data.txt', 1, X, Y)
    X, Y = read_data('/mnt/ssd/Repos/DNA-Sequence-Machine-learning/dog_data.txt', 2, X, Y)

    print(np.array(X).shape, np.array(Y).shape)
    
    #gene(X, Y, 3)
    #gene(X, Y, 6)
    gene(X, Y, 9)
    gene(X, Y, 12)

    #gene_species2(X, Y)
    #gene_species(X, Y)
    #species(X, Y)


# - - - - - - - RUN - - - - - - - 
if __name__ == "__main__":
    main()