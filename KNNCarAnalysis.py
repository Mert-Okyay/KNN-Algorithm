import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn import preprocessing as pp

data = pd.read_csv("car.data")
print (data.head())

le = pp.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls=data["class"].map({"unacc": 0, "acc": 1,"good": 2,"vgood": 3})

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

parameter = 100.
bestOne = 0
for neighbourCount in range(1, 31):
    print("Neightbour Count: ", neighbourCount)
    model = KNeighborsClassifier(n_neighbors=neighbourCount)

    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print("Accuracy: ", accuracy)

    predicted = model.predict(x_test)
    names = ["unacc", "acc", "good", "vgood"]

    errorCount = 0
    for d in range(len(x_test)):
        if (predicted[d]!=y_test[d]):
            indicator = "Inaccurate"
            errorCount = errorCount+1
        else:
            indicator = "Accurate"
        ##print("Predicted: ", names[predicted[d]], "Data: ", x_test[d], "Actual: ", names[y_test[d]]," ", indicator)

    print("Number of Errors: ", errorCount)
    if errorCount < parameter:
        parameter = errorCount
        bestOne = neighbourCount

print("Best case: ", bestOne)