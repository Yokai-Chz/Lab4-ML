'''
Hernández Jiménez Erick Yael
Patiño Flores Samuel
Robert Garayzar Arturo

Descripción: Programa que evalua los diferentes meétodos de validación utilizados en machine learning
Toma dos datasets (Iris y Wine), los cuales en contraste uno es un dataset equilibrado y otro es uno más
desbalanceado. Para notar la diferencia de implementación de los métodos de validación. Mostrando con gráficas
las diferentes precisiones obtenidas al aplicarlas al modelo KNN

'''



# Importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.datasets import load_iris, load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# Función para cargar datasets
def load_dataset(choice):
    if choice == 1:
        data = load_iris()  # Dataset Iris (balanceado)
        X, y = data.data, data.target
        dataset_name = "Iris"
    elif choice == 2:
        data = load_wine()  # Dataset Wine (desbalanceado)
        X, y = data.data, data.target
        dataset_name = "Wine"
    else:
        raise ValueError("Opción de dataset inválida. Selecciona 1 o 2.")

    # Contamos la distribución de clases
    class_counts = Counter(y)
    print(f"Dataset seleccionado: {dataset_name}")
    print(f"Distribución de clases: {class_counts}")

    return X, y, dataset_name, class_counts

# Hold-Out Validation
def hold_out_validation(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# K-Fold Cross Validation
def k_fold_validation(X, y, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy

# Leave-One-Out Cross Validation
def leave_one_out_validation(X, y):
    loo = LeaveOneOut()
    accuracies = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy

# Función para realizar todas las validaciones y almacenar resultados
def run_validations(X, y, test_size, k):
    # Hold-Out Validation
    hold_out_acc = hold_out_validation(X, y, test_size)

    # K-Fold Validation
    k_fold_acc = k_fold_validation(X, y, k)

    # Leave-One-Out Validation
    loo_acc = leave_one_out_validation(X, y)

    return hold_out_acc, k_fold_acc, loo_acc

# Graficar los resultados de validación
def plot_results(accuracies):
    methods = ['Hold-Out', 'K-Fold', 'Leave-One-Out']

    fig, ax = plt.subplots()

    x = np.arange(len(methods))
    width = 0.35  # Ancho de las barras

    rects1 = ax.bar(x - width/2, accuracies['Iris'], width, label='Iris')
    rects2 = ax.bar(x + width/2, accuracies['Wine'], width, label='Wine')
    print("Precisión Iris:")
    print(accuracies['Iris'])
    print("Precisión Wine:")
    print(accuracies['Wine'])

    ax.set_ylabel('Precisión')
    ax.set_title(f'Precisión por método de validación y dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    fig.tight_layout()
    plt.show()

# Graficar desbalance de clases
def plot_class_distribution(class_counts, dataset_name):
    labels = class_counts.keys()
    values = class_counts.values()

    plt.bar(labels, values)
    plt.title(f'Distribución de clases en {dataset_name}')
    plt.xlabel('Clases')
    plt.ylabel('Número de muestras')
    plt.show()

# Programa principal
def main():
    print("Ejecutando validaciones en los datasets Iris y Wine...")

    # Parámetros de validación
    test_size = float(input("Ingrese el tamaño de prueba (ejemplo: 0.2 para 20%): "))
    k_folds = int(input("Ingrese el número de pliegues para K-Fold: "))

    # Cargar datasets
    X_iris, y_iris, name_iris, iris_class_counts = load_dataset(1)
    X_wine, y_wine, name_wine, wine_class_counts = load_dataset(2)

    # Mostrar desbalance de clases
    plot_class_distribution(iris_class_counts, name_iris)
    plot_class_distribution(wine_class_counts, name_wine)

    # Resultados para el dataset Iris
    iris_hold_out_acc, iris_k_fold_acc, iris_loo_acc = run_validations(X_iris, y_iris, test_size, k_folds)

    # Resultados para el dataset Wine
    wine_hold_out_acc, wine_k_fold_acc, wine_loo_acc = run_validations(X_wine, y_wine, test_size, k_folds)

    # Almacenar resultados en un diccionario
    accuracies = {
        'Iris': [iris_hold_out_acc, iris_k_fold_acc, iris_loo_acc],
        'Wine': [wine_hold_out_acc, wine_k_fold_acc, wine_loo_acc]
    }

    # Graficar resultados
    plot_results(accuracies)

if __name__ == "__main__":
    main()
