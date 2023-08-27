#!/usr/bin/env python
# coding: utf-8

# ## Renata Vargas - A01025281
# Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


# ## Parte 1: Lecutura de datos

# In[4]:


# Leemos datos
df = pd.read_csv('Salary_dataset.csv')  

# Asignamos X y Y
X = df['YearsExperience'].values
y0 = df['Salary'].values

# Graficamos los datos
plt.scatter(X, y0, marker='o', c='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X and Y')
plt.show()


# ## Parte 2: Modelo Regresión Lineal

# In[5]:


# Definimos el modelo de Regresión Lineal

def update_w_and_b(X, y, w, b, alpha):
  '''Update parameters w and b during 1 epoch'''
  dl_dw = 0.0
  dl_db = 0.0
  N = len(X)
  for i in range(N):
    dl_dw += -2*X[i]*(y[i] - (w*X[i] + b))
    dl_db += -2*(y[i] - (w*X[i] + b))
  # update w and b
  w = w - (1/float(N))*dl_dw*alpha
  b = b - (1/float(N))*dl_db*alpha
  return w, b

def train(X, y, w, b, alpha, epochs):
  '''Loops over multiple epochs and prints progress'''
  print('Training progress:')
  for e in range(epochs):
    w, b = update_w_and_b(X, y, w, b, alpha)
  # log the progress
    if e % 400 == 0:
      avg_loss_ = avg_loss(X, y, w, b)
      # print("epoch: {} | loss: {}".format(e, avg_loss_))
      print("Epoch {} | Loss: {} | w:{}, b:{}".format(e, avg_loss_, round(w, 4), round(b, 4)))
  return w, b

def train_and_plot(X, y, w, b, alpha, epochs, x_max_plot):
  '''Loops over multiple epochs and plot graphs showing progress'''
  for e in range(epochs):
    w, b = update_w_and_b(X, y, w, b, alpha)
  # plot visuals for last epoch
    if e == epochs-1:
      avg_loss_ = avg_loss(X, y, w, b)
      x_list = np.array(range(0,x_max_plot)) # Set x range
      y_list = (x_list * w) + b # Set function for the model based on w & b
      plt.scatter(x=X, y=y)
      plt.plot(y_list, c='r')
      plt.title("Epoch {} | Loss: {} | w:{}, b:{}".format(e, (avg_loss_,2), (w, 4), (b, 4)))
      plt.show()
  return w, b

def avg_loss(X, y, w, b):
  '''Calculates the MSE'''
  N = len(X)
  total_error = 0.0
  for i in range(N):
    total_error += (y[i] - (w*X[i] + b))**2
  return total_error / float(N)

def predict(x, w, b):
  return w*x + b


# ## Parte 3: Evaluación Modelo

# In[6]:


# Inicializamos variables
w = 0.0
b = 0.0
alpha = 0.01
epochs = 12000

# Llamamos al modelo 
w, b = train(X=X, y=y0, w=0.0, b=0.0, alpha=0.001, epochs=epochs)


# In[7]:


# Definimos algunos epochs para que se grafiquen esos
epoch_plots = [1, 2, 3, 11, 51, 101, epochs+1]
for epoch_plt in epoch_plots:
  w, b = train_and_plot(X, y0, 0.0, 0.0, 0.001, epoch_plt, 50)


# ## Parte 4: Mejoramiento del modelo

# In[8]:


# Cross Validation 

# Definición de datos 
X = X
y = y0

# Definimos las variables necesarias 
num_folds = 5
num_epochs = 1000
learning_rate = 0.001
patience = 10

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

best_model = None
best_loss = float('inf')

for fold in range(num_folds):
    fold_size = len(X) // num_folds
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size
    
    X_train = np.concatenate([X[:val_start], X[val_end:]])
    y_train = np.concatenate([y[:val_start], y[val_end:]])
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    
    w = np.random.randn(1)
    b = np.random.randn(1)
    
    counter = 0
    
    for epoch in range(num_epochs):
        # Calcula las predicciones
        y_pred = w * X_train + b
        
        # Calcula gradientes
        dw = -2 * np.mean((y_train - y_pred) * X_train)
        db = -2 * np.mean(y_train - y_pred)
        
        # Actualiza parámetros
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Calula la pérdida 
        y_val_pred = w * X_val + b
        val_loss = mean_squared_error(y_val, y_val_pred)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_w = w
            best_b = b
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    if best_model is None:
        best_model = (best_w, best_b)
    
    w, b = best_model
    
    # Calcula la pérdida del test por fold
    y_test_pred = w * X_val + b
    test_loss = mean_squared_error(y_val, y_test_pred)
    print(f"Test loss for fold {fold}: {test_loss}")


# In[9]:


# Definimos función para entrenar y regresar el progreso (valor de loss)
def train(X, y, w, b, alpha, epochs):
  '''Loops over multiple epochs and prints progress'''
  print('Training progress:')

  #Inicializamos lista y variable
  avg_loss_list = []
  loss_last_epoch = 9999999

  # Entre cada epoch, actualiza los valores de w y b con ayuda de la función update_w_and_b() y calcula el loss
  for e in range(epochs):
    w, b = update_w_and_b(X, y, w, b, alpha)
    avg_loss_ = avg_loss(X, y, w, b)

  
  # Imprime el progreso cada 400 epochs
    #if e % 10 == 0:
      # print("epoch: {} | loss: {}".format(e, avg_loss_))
    print("Epoch {} | Loss: {} | w:{}, b:{}".format(e, avg_loss_, (w, 4), (b, 4)))
    
    avg_loss_list.append(avg_loss_) #añade el loss a una lista
    loss_step = abs(loss_last_epoch - avg_loss_) #* Calcula la diferencia entre el ultimo loss y el actual
    loss_last_epoch = avg_loss_ #* actualiza el valor del ultimo loss al actual para la siguiente epoch
    
    # Para de entrenar si la diferencia del error en el epoch anterior contra el epoch actual es menor a 0.001
    if loss_step < 0.0001: #*
      print('\nStopping training on epoch {}/{}, as (last epoch loss - current epoch loss) is less than 0.001 [{}]'.format(e, epochs, loss_step)) #*
      break #*

  return w, b # Regresa los últimos valores de w y b


# In[10]:


nuevo = train (X, y0, w, b, 0.001, epochs)


# ## Parte 5: Comparación con Sklearn

# Por fines comparativos, utilizaré librerías predeterminadas

# In[11]:


# Importamos libreríaaaas <3 :D 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Importamos datos y definimos variables
X = df['YearsExperience'].values
y = df['Salary'].values

X = X.reshape(-1, 1)

# Dividimos en sets de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Llamamos al modelo de Regresión
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Graficamos para ver que tan chido salió 
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.legend()
plt.show()

#Accuracy 
r2_score = model.score(X_test, y_test)
print(f"R-squared Score: {r2_score}")


# ## Parte 6: Reflexión y análisis

# En esta práctica logramos aplicar un modelo de regresión lineal sin librerías con el fin de entender como funciona este tipo de modelo en Machine Learning. Lo que pondremos a prueba de este modelo es como podemos ajustar de manera más precisa una función a los datos obtenidos con el fin de hacer una predicción con nuevos potenciales datos. Al hacer un análisis de Regresión Lineal estamos asumiendo que los datos siguen una distribución normal, linealidad, homoscedasticidad, independencia y no multicolinealidad (Zack West - 2018), lo cual haremos en este caso. 
# 
# Para este caso, utilicé una base de datos que contenía dos variables: Salario y Años de Experiencia. Es una base de datos simple para poder entender este tipo de regresión (En caso de utilizar bases de datos con múltiples variables, esta se vuelve una Regresión Lineal Múltiple). Se puede observar en la primera parte que decidí que la variable a predecir iba a ser Salarios, por lo cual esa fue mi Y y Años de Experiencia fue mi X. La adaptación de la función se busca hacer en el set de entrenamiento y será evaluada con el set de prueba. 
# 
# En la segunda parte del código desglozamos la representación matemática del modelo: la Función de Hipótesis representada por Y = b_0 + b_1 X + e. Donde Y son todos los valores de variable dependiente, b_0 la intersección (que funciona como el bias), b_1 es el coeficiente (este determinara la pendiente de la función) y por último e es el error. En este caso utilizaremos el Error Cuadrático Medio (mejor conocido como MSE en inglés). El objetivo es que la distancia (medida por el MSE) sea la menor posible con la predicción y los valores reales y que esto resulte en una correlación fuerte (puede ser positiva o negativa, pero entre más cercana a ). Esto se lleva a cabo con el gradiente descendiente: este funciona con derivadas que ayudan a determinar la dirección en la que se deben de ajustar los parámetros del modelo para que sean más eficientes. 
# 
# En la tercera parte del código, evaluamos al modelo en donde obtuvimos una disminución importante del error conforme avanzaban los epochs. El cambio de parámetros se puede observar más drástico que el error ya que alrededor del epoch 6800, los cambios en el error comienzan a ser menos notorios. Recordemos que en este caso el error es la diferencia entre el valor predecido y el valor real, así que las pequeñas diferencias en realidad no serán de gran modificación al momento de evaluar el modelo. Como mencioné anteriormente, se observa un cambio en los parámetros que son los valores responsables de la posición y la pendiente de la Función de Hipótesis. Posterior a eso, se graficaron en algunos epochs, la función junto con algunos datos y conforme avanzaban más, se puede observar como el modelo ¡se une de manera exitosa a los datos! 
# 
# En la cuarta sección del código intenté hacer un mejoramiento del modelo con Cross Validation. Que es un método que separa a los datos en muestras aleatorias llamadas 'folds' donde por cada fold, se corre el modelo y se evalúan los parámetros a una escala mucho más pequeña. Los resultados fueron favorables para el primer fold porque era el error más pequeño, sin embargo no logré averiguar porque todos las vueltas se detenían en el epoch 9. Le voy a preguntar al profe el lunes. También repliqué el código de optimización de epochs. 
# 
# Por último, al final utilicé una librería que nos permite aplicar el modelo de manera sumamente fácil y eficiente para una comparación visual de lo fácil que es hacerlo con las librerías que ya existen (y apreciarlas un poco más). 
# 
# 
