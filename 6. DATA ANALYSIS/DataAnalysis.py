"""
1.1. Leer csv sin encabezados
"""

import pandas as pd 
url = "https://............"
df = pd.read_csv(url, header=None)


"""
1.2. Ver las primeras n filas de un dataframe
"""

df.head(n)


"""
1.3. Ver las últimas n filas de un dataframe
"""

df.tail(n)

"""
1.4 Agregar texto de encabezados
"""

headers = [“encabezado1”, “enc. 2”, ….]
df.columns = headers


"""
1.5. Exportar CSV
"""

path = “C:/…….../nombreArchivo.csv”
df.to_csv(path)


"""
1.6. Tipos de cada columna en un dataframe
"""
df.dtypes


"""
1.7. Estadística descriptiva
"""
df.describe()


"""
1.8. Estadística descriptiva incluyendo los tipos de datos object
"""
df.describe(include="all")



"""
1.8. Conexión a base de datos
"""

 from dbmodule import connect
 # creo un objeto conexión
 connection = connect (‘databasename’,‘user’,’password’)
 # creo un objeto cursor
 cursor = connection.cursor()
 # ejecuto consultas
 cursor.execute(‘select * from mytable’)
 results = cursor.fetchall()
 # libero recursos
 cursor.close()
 connection.close()



"""
2.1. Acceder a columna de dataframe (symboling en el ejemplo)
"""

df["symboling"]


"""
2.2. Manipular columna del dataframe (sumo 1 en el ejemplo)
"""

df["symboling"] = df["symboling"] + 1



"""
2.3. Borrar columnas
"""
df.drop(["id","name"],axis=1,inline=True)


"""
2.3.1. Borrar datos faltantes
"""




df.dropna(subset=[“price”], axis=0,inplace=True)

#axis = 0: borra toda la fila
#axis = 1: borra toda la columna.
#inplace = True: modifica directamente el conjutno de datos.


"""
2.4. Remplazar datos faltantes (por la media en el ejemplo)
"""

mean = df[“normalized-losses”].mean()
df[“normalized-losses”].replaces(np.nan,mean)


"""
2.5. Renombrar una columna
"""

df.rename(columns = {“city-mpg”: “city-mpg/100kmh”}, inplace=True)


"""
2.6. Tipos de datos
"""

dataframe.dtypes


"""
2.7. Convertir tipo de datos (a entero en el ejemplo)
"""

df["price"] = df[‘price’].astype(“int”)


"""
2.8. Binning (3 bins en el ejemplo)
"""

bins = np.linspace(min(df[“price”]),max(df[“price”]), 4)
group_names = [“Low”, “Medium”, “High”]
df[“price-binned”] = pd.cut(df[“price”], bins,labels=group:names,include_lowest=True)


"""
2.9. Convertir variables categóricas en cuantitativas
"""

pd.get_dummies(df[‘fuel’])




"""
3.1. Resumir variables categóricas (se cambia el nombre por legibilidad)
"""

drive_wheels_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheels_counts.rename(columns={"drive-wheels":"value_counts"},inplace=True)

"""
3.2. Diagrama de caja (drive-wheels sobre price)
"""

sns.boxplot(x="drive-wheels", y="price", data=df)



"""
3.3. Gráfico de dispersión (para variables continuas)
"""

%matplotlib

y = df["price"]
x = df["engine-size"]
plt.scatter(x,y)

plt.title("scatterplot of engine-size vs price")
plt.xlabel("engine-size")
plt.ylabel("Price")


"""
3.4. Group by 

Puede agrupar mediante una única variable o mediante un grupo.
Como ejemplo digamos que estamos interesados en encontrar el precio promedio de vehículos y observar cómo difieren entre los diferentes estilos de cuerpo (body styles) y ruedas de manejo (drive wheels).
Para hacer esto primero elegimos las 3 columnas de datos en las que estamos interesados. Luego agrupamos los datos reducidos acorde a “drive wheels” y “body styles”.
Dado que estamos interesados en saber cómo difiere el precio promedio en todos los ámbitos, podemos tomar la media de cada grupo y agregarlo también al final de la línea.
Los datos están ahora agrupados en subcategorías y solamente el precio promedio de cada subcategoría se muestra.
"""


df_test = df[["drive-wheels", "body-style", "price"]]
df_grp = df_test.groupby[["drive-wheels", "body-style"], as_index=False]].mean()


"""
3.5. Tabla pivote
"""

df_pivot = df_grp.pivot(index="drive-wheels", columns="body-style")

"""
3.6. Mapa de calor
"""

plt.color(df_pivot, cmap="RdBu")
plt.colorbar()
plt.show()


"""
3.7. Diagrama de dispersión con Seaborn, pueden observrase las correlaciones
"""

sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)


"""
3.8. Correlaciòn de Pearson, SciPy
"""

pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])


"""
3.9. ANOVA
"""

df_anova = df[["make","price"]]
grouped_anova = df_anova.groupby(["make"])


"""
4.1. Regresión Lineal
"""

from sklearn.linear_model import LinearRegression
# creo un objeto regresiòn lineal usando el constructor
lm = LinearRegression()


"""
4.2. Ajustando y realizando predicciones con el modelo de regresión lineal
entrada: características y target
salida: arreglo con las predicciones
los parámetros son atributos
"""

# definimos variable predictora y objetivo
X = df [["highway-mpg"]]
Y = df[["price"]]
# ajustamos el modelo (obtebemos los parámetros b0 y b1 en reg. lineal)
lm.fit(X,Y)
# obtenemos una predicciòn
Yhat = lm.predict(X)
# parámetros
lm.intercept_ # bo
lm.coef_ # b1 (pte)


"""
4.3. Regresión Lineal Múltiple
"""

# almacenamos los 4 predictores en la variable Z
Z = df[["horsepower","curb-weight","engine-size","highway-mpg"]]
# entrenamos el modelo
lm.fit(Z,df["price"])
# obtenemos una predicciòn
Yhat = lm.predict(X)
# intercepciòn (b0)
lm.intercept_
# coeficientes (b1, b2, b3, b4)
lm.coef_

"""
4.4. Gráfico de regresiòn con seaborn
"""

import seaborn as sns
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

"""
4.5. Gráfico residual con seaborn
"""

import seaborn as sns
sns.residplot(df["highway-mpg", df["price"]])


"""
4.6. Gráfico de dist. con seaborn
"""

import seaborn as sns

ax1 = sns.displot(df["price"], hist=False, color="r", label="Valor verdadero")

sns.displot(Yhat, hist=False, color="r", label="Fitted values", ax=ax1)



"""
4.7. Modelo de regresión polinómica
"""

# polinomio de tercer orden
f = np.polytfit(x,y,3)
p = np.polyld(f)
# imprimimos el modelo
print(p) 

"""
4.8. Regresiòn  polinómica multidimensional
"""

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2, include_bias=False)



"""
4.9. Transformaciòn de datos
"""

pr = PolynomialFeatures(degree=2)
pr.fit_transform([1,2], include_bias=False)

"""
4.10. Escalar característica simultáneamente
"""

from sklearn.preprocessing import StandardScaler

SCALE = StandardScaler()
SCALE.fit(x_data[["horsepower","highway-mpg"])
x_scale = SCALE.transform(x_data[["horsepower","highway-mpg"]])


"""
4.11.
Simplificamos el proceso usando una tubería. Pipeline realiza secuencialmente una serie de transformaciones. El último paso lleva a cabo una predicción. 
    • Importamos todos los módulos que necesitamos.
    • Creamos una lista de tuplas, el primer elemento de la tupla contiene el nombre del modelo estimador. 
    • El segundo elemento contiene el constructor del modelo. 
    • Ingresamos la lista en el constructor de la tubería. 
    • Ahora tenemos un objeto pipeline. 
    • Podemos entrenar el pipeline aplicando el método de entrenamiento al objeto de pipeline. 
    • También podemos producir una predicción. 
    • El método normaliza los datos, realiza una transformación polinómica, luego genera una predicción.
"""

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from slearn.pipeline import Pipeline

Input = [ ("Scale", StandardScaler()), ("polynomial", PolynomialFeatures(degree=2),...... ("mode",LinearRegression()))]
pipe = Pipeline(Input) # objeto pipeline

Pipe.fit(df[["horsepower","curb-weight","engine-size","highway-mpg"]], y)

yhat = Pipe.predict(X[["horsepower","curb-weight","engine-size","highway-mpg"]])

#pipe.score para el R2






"""
4.12. MSE
tiene dos entradas: el valor real de la variable objetivo y el valor predicho de la variable objetivo. 
"""

from sklearn.metrics import mean_squared_error

mean_squared_error(df["price"], Y_predict_simple_fit)


"""
4.13. R cuadrado
"""

X = df[["highway-mpg"]]
Y = df["price"]

lm.fit(X, Y)


"""
4.14.
Veamos un ejemplo de predicción. 
Si recuerda, entrenamos el modelo usando el método de ajuste (fit). 
Ahora queremos averiguar cuál sería el precio para un coche que tiene una autopista millas por galón de 30. 
"""

# entrenamos el modelo
lm.fit(df["highway-mpg", df["prices"])
# predecimso el precio para un auto con 30 highway-mpg
lm.predict(np.array(30.0).reshape(-1,1))
# coef.
lm.coef_

"""
4.15.
Secuencia de valores en un rabgo
"""

import numpy as np
# generamos una secuencia de 1 a 100
new_input = np.arrange(1,101,1).reshape(-1,1)


"""
5.1. Separando datos de test y de entrenamiento
"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)
test size es el % para testing



"""
5.2.
VALIDACIÓN CRUZADA
La función devuelve una matriz de puntuaciones, una para cada partición que se eligió como el conjunto de pruebas. 
Podemos promediar el resultado juntos para estimar de muestra r al cuadrado usando la función mean de NumPy. 


La forma más sencilla de aplicar la validación cruzada es llamar a la función cross_val_score, que realiza múltiples evaluaciones fuera de muestra. Este método se importa del paquete de selección de modelo de sklearn. Luego usamos la función cross_val_score. 
Los primeros parámetros de entrada son:
    - El tipo de modelo que estamos utilizando para hacer la validación cruzada. 
        -En este ejemplo, inicializamos un modelo de regresión lineal u objeto lr que pasamos a la función cross_val_score. 
    - x_data: los datos de la variable predictiva.
    - y_data: los datos de la variable objetivo. 
    - Podemos gestionar el número de particiones con el parámetro cv. 
        - En el ejemplo cv es igual a tres, lo que significa que el conjunto de datos se divide en tres particiones iguales. 
"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, x_data, y_data, cv=3)
np.mean(scores)

"""
La función cross_val_score devuelve un valor de puntuación para indicarnos el resultado de la validación cruzada. 
¿Y si queremos un poco más de información?  ¿Qué pasa si queremos conocer los valores predichos reales suministrados por nuestro modelo antes de que se calculen los valores r cuadrados? 
Para ello, utilizamos la función cross_ val_predict. 
"""

from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict (lr2e, x_data, y_data, cv=3)


"""
5.2. Valores varios de R2
Podemos calcular diferentes valores R cuadrados de la siguiente manera. 
    - Creamos una lista vacía para almacenar los valores. 
    - Creamos una lista que contiene diferentes Ã³rdenes de polinomios. 
    - Iteramos a travÃ©s de la lista usando un bucle. 
    - Creamos un objeto de entidad polinómica con el orden del polinomio como parÃ¡metro. 
    - Transformamos los datos de entrenamiento y pruebas en un polinomio utilizando el mÃ©todo de transformación de ajuste. 
    - Ajustamos el modelo de regresiÃ³n usando los datos de transformación. 
    - Calculamos el R cuadrado usando los datos de prueba y lo almacenamos en la matriz.

"""

Rsqu_test = []
order = [1,2,3,4]


for n in order:
  pr = PolynomialFeatures(degree=n)
  x_train_pr = pr.fit_transform(x_train[['horsepower']])
  x_test_pr = pr.fit_transform(x_test[['horsepower']])
  lr.fit(x_train_pr,y_train)
  Rsqu_test.append(lr.score(x_test_pr,y_test))




"""
5.4. Predicción usando regresión de Ridge
"""

from sklearn.lineal_model import Ridge
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(X,y)
Yhat = RdigeModel.predict(X)


"""
5.5. Grid search
"""


from sklearn.lineal_models import Ridge
from sklearn.model_selection import GridSearchCV

parameters1 = [{'alpha':[0.001,0.1,1]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, parameter1, cv=4)
Grid1.fit(x_data[['horsepower','curb-weight',.....]],y_data)
Grid1.best_estimator_
scores = Grid1.cv_results_
scores['mean_test_score']
#devuelve un array



"""
5.6. GRID SEARCH pudiendo normalizar
"""

from sklearn.lineal_models import Ridge
from sklearn.model_selection import GridSearchCV

parameters2 = [{'alpha':[0.001,0.1,1], 'normalize':[True,False]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, parameters2, cv=4)
Grid1.fit(x_data[['horsepower','curb-weight',.....]],y_data)
Grid1.best_estimator_
scores = Grid1.cv results


#Imprimo la puntación de los params libres de lo anterior
for param,mean_val,mean_test inzip(scores['params'], scores['mean_test_score'], scores['mean_train_score']):
	print(param, "R2 on train data", mean_test)







































































