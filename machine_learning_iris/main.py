import pandas as pd #odczytywanie danych
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC #svm
from sklearn.ensemble import RandomForestClassifier  #rf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



# Wczytywanie danych z pliku CSV
data = pd.read_csv('Iris.csv')

data = data.drop(columns= ['Id'])


# Definiowanie danych wejściowych (X) i etykiet (Y)
X = data.drop('Species', axis=1)

Y = data['Species']

# Podział danych na zbiór treningowy i testowy (70% treningowy, 30% testowy)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# knn
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print("Dokładność modelu KNN: ",model.score(X_test, Y_test) * 100)

# Przewidywanie na zbiorze testowym
expected = Y_test
predicted = model.predict(X_test)

#Wyświetlanie metryki ocen danych
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))

"""
# SVM
model = SVC()
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print("Dokładność modelu SVM: {:.2f}%".format(accuracy * 100))

# Przewidywanie na zbiorze testowym
expected = Y_test
predicted = model.predict(X_test)

#Wyświetlanie metryki ocen danych
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
"""

"""
# Random Forests
model = RandomForestClassifier()
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print("Dokładność modelu Random Forests: {:.2f}%".format(accuracy * 100))

# Przewidywanie na zbiorze testowym
expected = Y_test
predicted = model.predict(X_test)

#Wyświetlanie metryki ocen danych
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
"""

"""
#pętla testująca różne ilości sąsiadów dla modelu KNN
for k in range(1, 101):
    model = KNeighborsClassifier(n_neighbors=k)  # Tworzenie modelu KNN z k sąsiadami
    model.fit(X_train, Y_train)  # Trenowanie modelu na danych treningowych
    accuracy = model.score(X_test, Y_test)
    print(f'Dokładność modelu z k = {k}: {accuracy * 100:.2f}%')
"""

"""
# Tworzenie wykresu
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Wykres 1: Długość płatków (PetalLengthCm) i szerokość płatków (PetalWidthCm)
sns.scatterplot(x="PetalLengthCm", y="PetalWidthCm", data=data, hue="Species", markers=["o", "s", "D"], ax=axes[0])
axes[0].set_title("Wykres punktowy na podstawie długości płatków i szerokości płatków")
axes[0].set_xlabel("Długość płatków (cm)")
axes[0].set_ylabel("Szerokość płatków (cm)")
axes[0].legend(title="Gatunek")

# Wykres 2: Długość  kielicha (SepalLengthCm) i szerokość kielicha (SepalWidthCm)
sns.scatterplot(x="SepalLengthCm", y="SepalWidthCm", data=data, hue="Species", markers=["o", "s", "D"], ax=axes[1])
axes[1].set_title("Wykres punktowy na podstawie długości kielicha i szerokości kielicha")
axes[1].set_xlabel("Długość kielicha (cm)")
axes[1].set_ylabel("Szerokość kielicha (cm)")
axes[1].legend(title="Gatunek")

plt.tight_layout()
plt.show()
"""
