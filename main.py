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

# Normalización

from sklearn.preprocessing import StandardScaler

# Normalización
scaler = StandardScaler()
X = scaler.fit_transform(X)
