User: ok ahora has lo mismo de manera completa para un apredisaje intermedio damede detañe de lo que hacel codigo tipo guia
Kimi: # Guía de Aprendizaje Intermedio de Machine Learning

## Introducción
Esta guía está diseñada para aquellos que ya tienen conocimientos básicos de Machine Learning y buscan profundizar en conceptos más avanzados. Vamos a explorar técnicas intermedias utilizando las mismas bibliotecas que en la guía básica, pero con mayor profundidad y ejemplos más complejos.

## 1. Optimización de Hiperparámetros con Grid Search

### Ejemplo: Optimización de un modelo de árboles de decisión
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Crear un modelo base
modelo = DecisionTreeClassifier()

# Definir la malla de hiperparámetros a probar
parametros = {
    'max_depth': [3, 5, 7, None],  # Profundidad máxima del árbol
    'min_samples_split': [2, 5, 10],  # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4]  # Mínimo de muestras en una hoja
}

# Crear el buscador de hiperparámetros
grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5)

# Ejecutar la búsqueda (X_train y y_train son datos previamente definidos)
grid_search.fit(X_train, y_train)

# Ver los mejores hiperparámetros encontrados
print("Mejores hiperparámetros:", grid_search.best_params_)

# Ver el rendimiento del mejor modelo
print("Mejor precisión:", grid_search.best_score_)
```

### Explicación línea por línea:
- **GridSearchCV**: Esta función crea un objeto que realizará una búsqueda exhaustiva sobre los parámetros especificados.
- **param_grid**: Es un diccionario donde las claves son nombres de parámetros y los valores son listas de valores a probar.
- **cv=5**: Especifica que se utilizará validación cruzada con 5 pliegues (folds).
- **fit**: Ejecuta la búsqueda de hiperparámetros en los datos de entrenamiento.
- **best_params_**: Atributo que contiene los hiperparámetros que dieron el mejor rendimiento.
- **best_score_**: Atributo que indica la mejor puntuación obtenida durante la búsqueda.

## 2. Árboles de Decisión para Clasificación

### Ejemplo: Clasificador de árboles de decisión
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Crear el modelo
modelo = DecisionTreeClassifier(max_depth=3)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Visualizar el árbol
plt.figure(figsize=(20,10))
plot_tree(modelo, feature_names=X.columns, class_names=['Clase A', 'Clase B'], filled=True)
plt.show()

# Realizar predicciones
predicciones = modelo.predict(X_test)
```

### Explicación línea por línea:
- **DecisionTreeClassifier**: Clase para crear un modelo de árbol de decisión.
- **max_depth=3**: Limita la profundidad máxima del árbol para evitar sobreajuste.
- **fit**: Entrena el modelo utilizando los datos de entrenamiento.
- **plot_tree**: Función para visualizar gráficamente la estructura del árbol de decisión.
- **feature_names**: Lista de nombres de las características para mostrar en el gráfico.
- **class_names**: Lista de nombres de las clases objetivo para mostrar en el gráfico.
- **predict**: Realiza predicciones en los datos de prueba.

## 3. Máquinas de Vector Soporte (SVM) para Clasificación

### Ejemplo: Clasificador SVM
```python
from sklearn.svm import SVC

# Crear el modelo
modelo = SVC(kernel='rbf', C=1.0, gamma='scale')

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Realizar predicciones
predicciones = modelo.predict(X_test)
```

### Explicación línea por línea:
- **SVC**: Clase para crear una máquina de vector soporte para clasificación.
- **kernel='rbf'**: Especifica el tipo de kernel a utilizar ('rbf' es el kernel de función base radial).
- **C=1.0**: Parámetro de regularización que controla el trade-off entre errores de entrenamiento y complejidad del modelo.
- **gamma='scale'**: Parámetro que define la influencia de cada ejemplo de entrenamiento.
- **fit**: Entrena el modelo utilizando los datos de entrenamiento.
- **predict**: Realiza predicciones en los datos de prueba.

## 4. Análisis de Componentes Principales (PCA) para Reducción de Dimensionalidad

### Ejemplo: Reducción de dimensionalidad con PCA
```python
from sklearn.decomposition import PCA

# Crear el modelo
pca = PCA(n_components=0.95)  # Mantener el 95% de la varianza

# Ajustar el modelo y transformar los datos
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Ver la varianza explicada por cada componente
print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
```

### Explicación línea por línea:
- **PCA**: Clase para realizar análisis de componentes principales.
- **n_components=0.95**: Especifica que se deben mantener suficientes componentes para explicar el 95% de la varianza.
- **fit_transform**: Ajusta el modelo PCA a los datos de entrenamiento y los transforma.
- **transform**: Transforma los datos de prueba utilizando el modelo ajustado.
- **explained_variance_ratio_**: Atributo que contiene la proporción de varianza explicada por cada componente principal.

## 5. Vecinos más Cercanos (KNN) para Clasificación

### Ejemplo: Clasificador KNN
```python
from sklearn.neighbors import KNeighborsClassifier

# Crear el modelo
modelo = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Realizar predicciones
predicciones = modelo.predict(X_test)
```

### Explicación línea por línea:
- **KNeighborsClassifier**: Clase para crear un modelo de vecinos más cercanos.
- **n_neighbors=5**: Especifica el número de vecinos a considerar para la clasificación.
- **weights='distance'**: Especifica que se deben ponderar las contribuciones de los vecinos por su distancia.
- **fit**: Entrena el modelo utilizando los datos de entrenamiento.
- **predict**: Realiza predicciones en los datos de prueba.

## Ejercicios para Practicar
1. Utiliza Grid Search para optimizar los hiperparámetros de un modelo de regresión lineal.
2. Crea un modelo de árboles de decisión para clasificar tipos de flores utilizando el dataset de iris.
3. Implementa un clasificador SVM para distinguir entre imágenes de dígitos escritos a mano.
4. Aplica PCA a un dataset de características de vinos para reducir su dimensionalidad.
5. Prueba diferentes valores de 'n_neighbors' en un modelo KNN para ver cómo afecta el rendimiento.

## Conclusión
Esta guía intermedia te ha proporcionado una introducción a técnicas más avanzadas de Machine Learning. Ahora puedes explorar aún más estos temas y combinarlos con los conceptos básicos que ya conoces para resolver problemas más complejos. Recuerda que la práctica continua es esencial para dominar estas técnicas.
