import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import time

#Start liczenai czasu
start_time = time.time()
#Wczytanie danych
#data = pd.read_csv('diabetes.csv', delimiter=';')
data = pd.read_csv('data1.csv', delimiter=';')
#data = pd.read_csv('data2.csv', delimiter=';')

#Podzial bazy danych ze wzgladu na atrybut decyzyjny

X = data.drop('Target', axis=1)
y = data['Target']
#diabetes
#X = data.drop('Mikroalbuminuria', axis=1)
#y = data['Mikroalbuminuria']
#data2
#X = data.drop('verification.result', axis=1)
#y = data['verification.result']


#Podzial na zbior testowy i treningowy
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.33, random_state=5)

#Wybranie funkcji dzielacej
splitting_criterion = 'gini'  # Opcje: 'gini', 'entropy'


#Ustawianie zmiennych drzewa
splitter = "best"  # Opcje: "best", "random"
max_depth = None  # Maksymalna glebokosc
min_samples_split = 2  # Minimalna ilosc przypadkow do rozdzielenia galezi
min_samples_leaf = 1  # minimalna ilosc przypadkow do utworzenia lisci
min_weight_fraction_leaf = 0.0  # Minimalna waga galezi aby utworzyc lisc
max_features = None  # Maksymalna ilosc przypadkow aby rozdzielic galaz
max_leaf_nodes = None  # Maksymalna ilosc lisci 

#Budowanie drzewa z podanymi parametrami
tree = DecisionTreeClassifier(
    criterion=splitting_criterion,
    splitter=splitter,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    min_weight_fraction_leaf=min_weight_fraction_leaf,
    max_features=max_features,
    max_leaf_nodes=max_leaf_nodes
)
tree.fit(X_train, y_train)

#Predykcje na zbiorze testowym oraz procent poprawnosci predykcji
y_pred = tree.predict(X_test)
predykcja = np.mean(y_pred == y_test)
print(f"Trafnosc: {predykcja}")


#Raport klasyfikacyjny
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
#wypisanie glebokosci drzewa iilosci lisci
glebokosc = tree.get_depth()
print(f"Glebokosc drzewa: {glebokosc}")
ilosc_lisci = tree.get_n_leaves()
print(f"ilosc lisci: {ilosc_lisci}")

#przygotowanie wyświetlania drzewa
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=X.columns, class_names=y.unique().astype(str), filled=True, rounded=True)

#Obliczenie czasu wykonywania programu i wypisanie go
end_time = time.time()
compilation_time = end_time - start_time
print("Czas działania:", compilation_time, "sekund")
#wyświetlenie drzewa
plt.show()