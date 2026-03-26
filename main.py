import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# Cargar dataset
df = pd.read_csv("data/ai4i2020.csv")  

# Inspeccionar dataset
print(df.head(5)) 
print(df.info())
print(df["Machine failure"].value_counts())

# Separar x - y
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Eliminar columnas irrelevantes
X = X.drop(["UDI", "Product ID"], axis=1)

# Codificar variable categórica
X["Type"] = X["Type"].map({"L": 0, "M": 1, "H": 2})

print(X.head())

# Dividimos los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalización
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODELO KNN

# Crear modelo
knn = KNeighborsClassifier(n_neighbors=7)

# Entrenar modelo
knn.fit(X_train, y_train)

# Predicciones
y_pred = knn.predict(X_test)

print(y_pred[:10])

# Precisión
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Informe completo
print(classification_report(y_test, y_pred))

# ÁRBOL DE DECISIÓN

tree = DecisionTreeClassifier(random_state=42)

# Entrenar modelo
tree.fit(X_train, y_train)

# Predicciones
y_pred_tree = tree.predict(X_test)

print(y_pred_tree[:10])

# Accuracy
acc_tree = accuracy_score(y_test, y_pred_tree)
print("Accuracy Árbol:", acc_tree)

# Reporte
print(classification_report(y_test, y_pred_tree))
