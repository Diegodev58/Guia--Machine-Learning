
---

# **Guía Completa para Aprender Machine Learning (Nivel Básico)**  
**Contenido:** NumPy, Pandas, Matplotlib/Seaborn, y algoritmos fundamentales de ML con Scikit-learn (Regresión, Clasificación, Clustering).  
**Objetivo:** Manipular datos, prepararlos y aplicar algoritmos básicos para resolver problemas cotidianos.

---

## **1. Manipulación de Datos con NumPy y Pandas**  
### **1.1. NumPy: Matemáticas con Arreglos**  
- **Propósito:** Trabajar con datos numéricos eficientemente.  
- **Funciones clave:**  
  - `.mean()`: Calcula el promedio.  
    ```python
    import numpy as np
    temperaturas = np.array([25, 28, 22, 30, 26])
    print(np.mean(temperaturas))  # Salida: 26.2
    ```
  - `.sum()`, `.min()`, `.max()`: Suma, mínimo y máximo.  
  - `.reshape()`: Cambiar la forma de un arreglo.  
    ```python
    ventas = np.array([100, 200, 300, 400]).reshape(2, 2)
    print(ventas)  # Salida: [[100 200], [300 400]]
    ```

---

### **1.2. Pandas: Datos Estructurados**  
- **Propósito:** Manejar tablas (DataFrames) y datos no estructurados.  
- **Ejemplo cotidiano:** Analizar ventas de una tienda.  
- **Funciones clave:**  
  - `.head()`: Ver primeras filas.  
    ```python
    import pandas as pd
    ventas_df = pd.DataFrame({
        "producto": ["manzana", "banana", "uva"],
        "unidades": [50, 30, 20],
        "precio": [1.5, 0.8, 2.0]
    })
    print(ventas_df.head())  # Muestra las primeras 3 filas
    ```
  - `.mean()`: Promedio por columna.  
    ```python
    print(ventas_df["precio"].mean())  # Salida: 1.433...
    ```
  - `.fillna()`: Rellenar datos faltantes.  
  - `.groupby()`: Agrupar datos.  
    ```python
    ventas_diarias = ventas_df.groupby("producto").sum()
    print(ventas_diarias)  # Suma unidades y precios por producto
    ```

---

## **2. Visualización de Datos con Matplotlib y Seaborn**  
### **2.1. Matplotlib: Gráficos Básicos**  
- **Ejemplo:** Graficar ventas mensuales.  
  ```python
  import matplotlib.pyplot as plt
  meses = ["Enero", "Febrero", "Marzo"]
  ventas = [1500, 2000, 1800]
  plt.plot(meses, ventas, marker="o")
  plt.title("Ventas Mensuales")
  plt.xlabel("Mes")
  plt.ylabel("Ventas")
  plt.show()
  ```
- **Funciones clave:**  
  - `plt.plot()`: Gráficos de líneas.  
  - `plt.scatter()`: Gráficos de dispersión.  
  - `plt.bar()`: Gráficos de barras.

---

### **2.2. Seaborn: Visualizaciones Estadísticas**  
- **Ejemplo:** Distribución de calificaciones de estudiantes.  
  ```python
  import seaborn as sns
  calificaciones = [65, 70, 85, 90, 55, 80]
  sns.histplot(calificaciones, bins=5)
  plt.title("Distribución de Calificaciones")
  plt.show()
  ```
- **Funciones clave:**  
  - `sns.histplot()`: Histogramas.  
  - `sns.heatmap()`: Mapas de calor para correlaciones.  
  - `sns.pairplot()`: Gráficos de pares.

---

## **3. Algoritmos de Machine Learning con Scikit-learn**  

### **3.1. Regresión (Predicciones Continuas)**  
- **Ejemplo:** Predecir el precio de una casa.  
  ```python
  from sklearn.linear_model import LinearRegression
  # Datos: Tamaño del terreno (m²) vs. Precio
  X = [[100], [150], [200], [250]]  # Tamaño
  y = [300000, 450000, 600000, 750000]  # Precio
  modelo = LinearRegression()
  modelo.fit(X, y)
  print(modelo.predict([[180]]))  # Predicción para 180m²
  ```
- **Funciones clave:**  
  - `.fit()`: Entrenar el modelo.  
  - `.predict()`: Hacer predicciones.  
  - `.score()`: Accuracy del modelo.

---

### **3.2. Clasificación (Etiquetas Categóricas)**  
- **Ejemplo:** Clasificar correos como "spam" o "no spam".  
  ```python
  from sklearn.svm import SVC
  from sklearn.model_selection import train_test_split
  # Datos: Longitud del texto vs. Si es spam (1) o no (0)
  X = [[100, 1], [200, 0], [150, 1], [50, 0]]  # Longitud + Etiqueta
  y = [1, 0, 1, 0]  # 1=spam, 0=no spam
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  modelo = SVC()
  modelo.fit(X_train, y_train)
  print(modelo.predict([[120, 1]]))  # Predicción para un nuevo correo
  ```
- **Funciones clave:**  
  - `train_test_split()`: Dividir datos en entrenamiento y prueba.  
  - `.accuracy_score()`: Medir precisión.

---

### **3.3. Clustering (Agrupamiento)**  
- **Ejemplo:** Segmentar clientes por hábitos de compra.  
  ```python
  from sklearn.cluster import KMeans
  # Datos: Gasto mensual vs. Visitas a la tienda
  X = [[500, 10], [200, 5], [800, 20], [300, 8]]
  modelo = KMeans(n_clusters=2)
  modelo.fit(X)
  print(modelo.labels_)  # Etiquetas de cluster para cada cliente
  ```
- **Funciones clave:**  
  - `KMeans()`: Algoritmo de agrupamiento.  
  - `.labels_`: Etiquetas asignadas a cada dato.

---

## **4. Pasos para Resolver un Problema de ML**  
1. **Definir el objetivo**: ¿Predicción? ¿Clasificación?  
2. **Preparar los datos**:  
   - Limpieza (rellenar valores faltantes con `.fillna()`).  
   - Escalar características (ej: `StandardScaler()` en Scikit-learn).  
3. **Seleccionar el algoritmo**: Regresión, Clasificación o Clustering.  
4. **Entrenar y evaluar**: Usar `.fit()` y métricas como `.score()`.  
5. **Interpretar resultados**: ¿El modelo es preciso? ¿Qué mejoras se pueden hacer?

---

## **5. Ejercicios para Practicar**  
1. **NumPy/Pandas**: Calcular el promedio de ventas de un producto.  
2. **Matplotlib**: Graficar el rendimiento de estudiantes en un curso.  
3. **Regresión**: Predecir el precio de un automóvil usando su año de fabricación.  
4. **Clasificación**: Crear un modelo que clasifique frutas por peso y color.  
5. **Clustering**: Agrupar productos por popularidad y precio.

---

**¿Qué viene después?**  
La próxima guía profundizará en conceptos intermedios como:  
- Optimización de hiperparámetros (Grid Search).  
- Redes Neuronales básicas.  
- Análisis de componentes principales (PCA).  
- Validación cruzada y métricas avanzadas.

¡Comienza practicando con pequeños datasets (ej: ventas de una tienda, calificaciones escolares) y ve escalando la complejidad! 🚀

### Ejemplos practios

User: sin ejemcios solo ejemplos praticos
Kimi: ¡Claro! Aquí tienes ejemplos prácticos y cotidianos para cada módulo sin explicaciones adicionales:

---

### **1. NumPy**
```python
import numpy as np

# Calcular el promedio de temperaturas semanales
temperaturas = np.array([25, 28, 22, 30, 26])
print(np.mean(temperaturas))  # Salida: 26.2

# Sumar ventas mensuales
ventas = np.array([100, 200, 150])
print(np.sum(ventas))  # Salida: 450

# Cambiar la forma de un arreglo (ej: ventas por semanas)
ventas_semanales = np.array([100, 150, 200, 250]).reshape(2, 2)
print(ventas_semanales)
# Salida:
# [[100 150]
#  [200 250]]
```

---

### **2. Pandas**
```python
import pandas as pd

# Crear un DataFrame de ventas de productos
ventas_df = pd.DataFrame({
    "producto": ["manzana", "banana", "uva"],
    "unidades": [50, 30, 20],
    "precio": [1.5, 0.8, 2.0]
})

# Ver las primeras filas
print(ventas_df.head())

# Calcular el precio promedio
print(ventas_df["precio"].mean())  # Salida: 1.433...

# Rellenar datos faltantes (ej: precio desconocido)
ventas_df["precio"].fillna(1.0, inplace=True)

# Agrupar ventas por producto
ventas_totales = ventas_df.groupby("producto").sum()
print(ventas_totales)
```

---

### **3. Matplotlib**
```python
import matplotlib.pyplot as plt

# Graficar ventas mensuales
meses = ["Enero", "Febrero", "Marzo"]
ventas = [1500, 2000, 1800]
plt.plot(meses, ventas, marker="o")
plt.title("Ventas Mensuales")
plt.xlabel("Mes")
plt.ylabel("Ventas ($)")
plt.show()

# Gráfico de dispersión: Precio vs. tamaño de casa
tamanos = [100, 150, 200, 250]
precios = [300000, 450000, 600000, 750000]
plt.scatter(tamanos, precios)
plt.title("Precio vs. Tamaño de Casa")
plt.xlabel("Tamaño (m²)")
plt.ylabel("Precio ($)")
plt.show()
```

---

### **4. Seaborn**
```python
import seaborn as sns

# Histograma de calificaciones de estudiantes
calificaciones = [65, 70, 85, 90, 55, 80]
sns.histplot(calificaciones, bins=5)
plt.title("Distribución de Calificaciones")
plt.show()

# Mapa de calor de una matriz de correlación
import numpy as np
matriz_correlacion = np.array([[1.0, 0.8], [0.8, 1.0]])
sns.heatmap(matriz_correlacion, annot=True)
plt.title("Correlación entre Variables")
plt.show()
```

---

### **5. Scikit-learn (Regresión)**
```python
from sklearn.linear_model import LinearRegression

# Predecir el precio de una casa por tamaño
X = [[100], [150], [200], [250]]  # Tamaño (m²)
y = [300000, 450000, 600000, 750000]  # Precio
modelo = LinearRegression()
modelo.fit(X, y)
print(modelo.predict([[180]]))  # Predicción para 180m²
```

---

### **6. Scikit-learn (Clasificación)**
```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Clasificar correos como spam (1) o no spam (0)
X = [[100, 1], [200, 0], [150, 1], [50, 0]]  # Longitud + Etiqueta
y = [1, 0, 1, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = SVC()
modelo.fit(X_train, y_train)
print(modelo.predict([[120, 1]]))  # Predicción para un nuevo correo
```

---

### **7. Scikit-learn (Clustering)**
```python
from sklearn.cluster import KMeans

# Segmentar clientes por gasto mensual y visitas
X = [[500, 10], [200, 5], [800, 20], [300, 8]]  # Gasto + Visitas
modelo = KMeans(n_clusters=2)
modelo.fit(X)
print(modelo.labels_)  # Etiquetas de cluster
```

---

### **8. Preprocesamiento con Scikit-learn**
```python
from sklearn.preprocessing import StandardScaler

# Escalar datos de ventas para modelos
ventas = [[100], [150], [200], [250]]
scaler = StandardScaler()
ventas_escaladas = scaler.fit_transform(ventas)
print(ventas_escaladas)
```

---

### **9. Evaluación de Modelos**
```python
from sklearn.metrics import accuracy_score

# Calcular precisión de un modelo de clasificación
y_pred = [1, 0, 1, 0]
y_true = [1, 1, 1, 0]
print(accuracy_score(y_true, y_pred))  # Salida: 0.75
```

---

### **10. División de Datos**
```python
from sklearn.model_selection import train_test_split

# Dividir datos en entrenamiento y prueba
X = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)  # Datos de entrenamiento
```

---

Estos ejemplos son completamente ejecutables y representan situaciones cotidianas (ventas, calificaciones, predicción de precios, etc.). ¡ Pruébalos en tu entorno de Python ! 🚀

