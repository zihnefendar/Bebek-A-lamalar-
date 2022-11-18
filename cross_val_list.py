import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

DATA_PATH = "library.json"
SAVE_PATH = "results.json"
results = {
    "segment": [],
    "classifier": [],
    "one_fit_score": [],
    "cross_val_score": [],
    "cross_val_high": [],
    "cross_val_deviation": []
}
names = [
    "SVM",
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Neural NET",
    "AdaBoost",

]

classifiers = [
    SVC(kernel="poly", C=22),
    KNeighborsClassifier(1),
    DecisionTreeClassifier(max_depth=50),
    RandomForestClassifier(max_depth=32, n_estimators=120, criterion='entropy'),
    MLPClassifier(alpha=0.1, max_iter=3000, hidden_layer_sizes=12, random_state=1, solver="lbfgs"),
    AdaBoostClassifier(n_estimators=100, random_state=0)
]

with open(DATA_PATH, "r") as fp:
    lib = json.load(fp)
print("Data succesfully loaded!")
for i in range(len(lib["segment"])):
    X = np.array(lib["data"][i]["mfcc"])
    X = X.reshape((X.shape[1] * X.shape[2]), X.shape[0])
    X = X.T
    y = np.array(lib["data"][i]["labels"])
    ns = lib["segment"][i]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    # iterate over classifiers
    print("---------------segment=", ns, "--------------------------")
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        # with open("models/" + name + ".pkl", "wb") as f:
        # pickle.dump(clf, f)

        score = clf.score(X_test, y_test)
        print(name, ": One fit score: %0.3f" % score)
        scores = cross_val_score(clf, X, y, cv=6)
        print(name, ": ", "%0.3f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print(scores)

        results["segment"].append(ns)
        results["classifier"].append(name)
        results["one_fit_score"].append(score)
        results["cross_val_score"].append(scores.mean())
        results["cross_val_high"].append(scores.max())
        results["cross_val_deviation"].append(scores.std())
with open(SAVE_PATH, "w") as fp:
    json.dump(results, fp, indent=4)
df = pd.read_json('results.json')
df.to_excel('results.xlsx')
