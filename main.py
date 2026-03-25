import pandas as pd

# Cargar dataset
df=pd.read_csv("data/ai4i2020.csv")  

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

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar modelo
knn.fit(X_train, y_train)

# Predicciones
y_pred = knn.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
