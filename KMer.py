import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import MultinomialNB as mnb

# - - - - - - - Methods - - - - - - - 
def kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

def check_predictions(x, y, classifier):
    y_pred = classifier.predict(x)
    accuracy, precision, recall, f1 = get_metrics(y, y_pred)
    print(pd.crosstab(pd.Series(y, name='Actual'), pd.Series(y_pred, name='Predicted')))
    print(f"\naccuracy = {accuracy:0.3f} \nprecision = {precision:0.3f} \nrecall = {recall:0.3f} \nf1 = {f1:0.3f}")


# - - - - - - - Logic - - - - - - - 
kmer_size = 6
outfolder = "/mnt/ssd/Data/Output/Examples"

example_sets = {
    "chimp": "/mnt/ssd/Repos/DNA-Sequence-Machine-learning/chimp_data.txt",
    "dog": "/mnt/ssd/Repos/DNA-Sequence-Machine-learning/dog_data.txt",
    "human": "/mnt/ssd/Repos/DNA-Sequence-Machine-learning/human_data.txt",
}

input = {}
output = {}
cv = CountVectorizer(ngram_range=(4,4))

for count, (key, value) in enumerate(example_sets.items()):
    print(f"{count} \t Working {key}")
    data = pd.read_table(value)
    input.setdefault(key, [])
    output.setdefault(key, [])
    for index, row in data.iterrows():
        #item = fq.kmer_encoding(kmers, row[0].lower())
        item = kmers_funct(row[0], kmer_size)
        item = " ".join(map(str, item))
        input[key].append(item)
        output[key].append(row[1])

    print(input[key][0])

    if count == 0:
        input[key] = cv.fit_transform(input[key])
    else:
        input[key] = cv.transform(input[key])

    print(input[key][0])

    data['class'].value_counts().sort_index().plot.bar( )
    plt.savefig(f"{outfolder}/{key}.png")
    print(f"\n{key} shape: {input[key].shape}")


x_train, x_test, y_train, y_test = tts(input["human"], 
                                        output["human"],
                                        test_size = 0.20,
                                        random_state = 42)

print(f"\nHuman Training Shape: {x_train.shape}")
print(f"Human Testing Shape: {x_test.shape}")

classifier = mnb(alpha=0.1)
classifier.fit(x_train, y_train)



print("\n\n\t\tHUMAN")
check_predictions(x_test, y_test, classifier)

print("\n\n\t\tCHIMP")
check_predictions(input["chimp"], output["chimp"], classifier)

print("\n\n\t\tDOG")
check_predictions(input["dog"], output["dog"], classifier)