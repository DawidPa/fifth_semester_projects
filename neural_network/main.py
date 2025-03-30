#biblioteki do 1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
#biblioteki do 2
from keras.models import Sequential
from keras.layers import Dense
#biblioteki do 4
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score


# 1.Przygotowanie danych: Zaimportuj zbiór danych MNIST i podziel go na zbiór treningowy i testowy. Znormalizuj wartości pikseli do zakresu od 0 do 1.

# Wczytaj dane treningowe i testowe z plików CSV
train_data = pd.read_csv('MNIST_CSV/mnist_train.csv')
test_data = pd.read_csv('MNIST_CSV/mnist_test.csv')

# Podziel dane na etykiety (Y) i cechy (X)
Y_train = train_data.iloc[:, 0].values
X_train = train_data.iloc[:, 1:].values

Y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values

# Normalizuj wartości pikseli do zakresu od 0 do 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Przekształć etykiety na postać one-hot encoding
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# 2. Zdefiniowanie architektury sieci neuronowej: Stwórz model sieci neuronowej, wybierając odpowiednią liczbę warstw i liczby
# neuronów w każdej warstwie. Możesz użyć biblioteki Keras lub TensorFlow do tworzenia modelu. Możesz użyć DNN, CNN czy RMDL.

model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))  # Warstwa wejściowa
model.add(Dense(64, activation='relu'))  # Warstwa ukryta
model.add(Dense(10, activation='softmax'))  # Warstwa wyjściowa

# 3. Trenowanie modelu: Skonfiguruj model, wybierając funkcję straty, optymalizator i metryki oceny.
# Trenuj model na zbiorze treningowym za pomocą algorytmu wstecznej propagacji błędu.
# Eksperymentuj z różnymi hiperparametrami, takimi jak współczynnik uczenia i liczba epok.

# Skompiluj model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Skonfiguruj hiperparametry
learning_rate = 0.001 #współczynnik uczenia
epochs = 10 #epoki
batch_size = 32 #rozmiar batcha

# Trenuj model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size)

# 4. Ocena modelu: Ocenić wydajność modelu na zbiorze testowym.
# Oblicz metryki, takie jak dokładność klasyfikacji, precyzja, czułość itp.

# Przewidywanie klas dla zbioru testowego
Y_pred = model.predict(X_test)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)

# Dokładność klasyfikacji
accuracy = accuracy_score(y_true, y_pred_classes)
print(f'Dokładność klasyfikacji: {accuracy * 100:.2f}%')

# Precyzja, czułość, F1-score
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = 2 * (precision * recall) / (precision + recall)

print(f'Precyzja: {precision:.2f}')
print(f'Czułość: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

# Raport klasyfikacji
class_report = classification_report(y_true, y_pred_classes)
print('Raport klasyfikacji:')
print(class_report)
