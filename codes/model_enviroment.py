import joblib
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from src.env.environment_model_config import arguments_parser
from src.preprocessing import (index_date, downcast_dtypes, nn_preparation,
                               lstm_preparation)
from src.env.environment_models import (create_lstm, create_nn,
                                        create_cnn1d)
from src.visualizations import (training_history, plot_multiple_xy_results,
                                watch_distributiions)
from src.utils import get_prime_factors, input_shape
from src.loss_functions import handler_loss_function
warnings.filterwarnings("ignore")
tf.random.set_seed(20)

# variables de entrenamiento
args = arguments_parser()
# lectura del dataframe
df = pd.read_csv('data/cleaned-data.csv')

# fechas inicial y final
fecha_inicial = df["fecha"].iloc[0][0: 10]
fecha_final = df["fecha"].iloc[-1][0: 10]
print("Fecha inicial de data de sensores: ", fecha_inicial)
print("Fecha final de data de sensores: ", fecha_final)

# buscar indice de finalizacion
end_date = datetime.strptime(df["fecha"].iloc[-1],
                             args.date_format) - timedelta(days=44)
end_date = end_date.strftime(args.date_format)

# número de timesteps con los que se va a trabajar
timesteps = args.timesteps
# separar los datos
test_index = df[df["fecha"] == end_date].index[0]
# a una freuencia de 15 minutos
days_of_prediction = len(df) - test_index
# start_date
start_date = df["fecha"].iloc[0][0: 10]
end_date = df["fecha"].iloc[test_index][0: 10]
# fechas del conjunto de testing
fechas = np.array(df["fecha"].iloc[test_index:])
# fecha como timestamp
df["fecha"] = df["fecha"].apply(
    lambda x: datetime.strptime(x, args.date_format))
# indexar fecha
df = index_date(df)
# bajar IOPS
df = downcast_dtypes(df)
# nombres
names = list(df.columns)
# reset de indices para eliminar las fechas
df.reset_index(drop=True, inplace=True)

# sacar datos targets columnas objetivo en la predicción
target_col = ['%cu_conc_final', '%cu_cola_final']

# datos de training
train_df = df.iloc[0: test_index]
# datos de testing con timesteps hacia atrás
test_df = df.iloc[test_index - timesteps:]

# vectores target
y_train = train_df[target_col].iloc[timesteps:, :].to_numpy()
y_test = test_df[target_col].iloc[timesteps:, :].to_numpy()
watch_distributiions(y_train, y_test, target_col)

# normalizar los datos
sc = MinMaxScaler(feature_range=(0, 1))
# training
train_df = sc.fit_transform(train_df)
# testing
test_df = sc.transform(test_df)
# guardar el scaler
joblib.dump(sc, "models/scaler.save")

# hacer reshape para las transformaciones de las celdas lstm
x_lstm_train, y_lstm_train = lstm_preparation(
    train_df, timesteps=timesteps)
x_lstm_test, y_lstm_test = lstm_preparation(test_df, timesteps=timesteps)

# separación por red neuronal fully connected
x_nn_train, y_nn_train = nn_preparation(
    train_df, names, target_col, timesteps=timesteps)
x_nn_test, y_nn_test = nn_preparation(
    test_df, names, target_col, timesteps=timesteps)

# reshape de data temporal
x_cnn1d_train =\
    x_lstm_train.reshape(-1, x_lstm_train.shape[1] * x_lstm_train.shape[2])
x_cnn1d_test =\
    x_lstm_test.reshape(-1, x_lstm_test.shape[1] * x_lstm_test.shape[2])

# concatenacion de estados anteriores + variables del estado actual
# x_cnn1d_train = np.concatenate((x_cnn1d_train, x_nn_train), axis=1)
# x_cnn1d_test = np.concatenate((x_cnn1d_test, x_nn_test), axis=1)

# data para la convolución 1D
x_cnn1d_train = x_cnn1d_train.reshape(
    x_cnn1d_train.shape[0], x_cnn1d_train.shape[1], 1)
x_cnn1d_test = x_cnn1d_test.reshape(
    x_cnn1d_test.shape[0], x_cnn1d_test.shape[1], 1)

# reshape para la cnn 2D
# numero de feautures
n_features = int(x_nn_train.shape[1])
# encuntra la factorización prima de los valores
prime_factorization = get_prime_factors(n_features)
shape = input_shape(prime_factorization,
                    natural_shape=train_df.shape[1])
# datos de la cnn
x_cnn_train = np.reshape(x_nn_train, (-1, shape[0], shape[1], 1))
x_cnn_test = np.reshape(x_nn_test, (-1, shape[0], shape[1], 1))

# crear el modelo multiinput
lstm_input_shape = (np.array(x_lstm_train).shape[1],
                    np.array(x_lstm_train).shape[2])
# timedistributed model
input_shape_conv_lstm = (None, x_cnn_train.shape[1], x_cnn_train.shape[2], 1)
# crear los distintos modelos a usar
nn = create_nn(x_nn_train.shape[1])
lstm = create_lstm(lstm_input_shape)
cnn1d = create_cnn1d((x_cnn1d_train.shape[1], 1))

# combinar las salidas
combined_input = tf.keras.layers.concatenate([cnn1d.output, lstm.output])

# continuar la concatenación de la red
x = tf.keras.layers.Dense(128)(combined_input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense(64)(combined_input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dense(y_train.shape[1], activation="linear")(x)

# generación del modelo siames
model = tf.keras.models.Model(
    inputs=[cnn1d.input, lstm.input], outputs=x)
print(model.summary())

# model = tf.keras.models.Model(
#     inputs=[nn.input, lstm.input, cnn1d.input], outputs=x)

print(model.summary())

# función de costos custom
tf.keras.losses.handler_loss_function = handler_loss_function

# model.compile(loss=args.loss,
#               optimizer=args.optimizer)
model.compile(loss=handler_loss_function(factor=1),
              optimizer=args.optimizer)

# llamar callbacks de early stopping
tf.keras.callbacks.Callback()
stop_condition = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=args.patience,
    verbose=1,
    min_delta=args.min_delta,
    restore_best_weights=True)

# bajar el learning_rate durante la optimización
learning_rate_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=args.lr_factor,
    patience=args.lr_patience,
    verbose=1,
    mode="auto",
    cooldown=0,
    min_lr=args.lr_min)

# cuales son los callbacks que se usaran
callbacks_ = [stop_condition, learning_rate_schedule]

# entrenar
history = model.fit(x=[x_cnn1d_train, x_lstm_train], y=y_train,
                    validation_split=args.validation_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    shuffle=True,
                    verbose=1,
                    callbacks=callbacks_)

training_history(history, model_name="Environment", filename="Model")

# evaluar el modelo
# results = model.evaluate([x_nn_test, x_lstm_test, x_cnn1d_test], y_test,
#                          batch_size=args.batch_size)
results = model.evaluate([x_cnn1d_test, x_lstm_test], y_test,
                         batch_size=args.batch_size)

# predictions
y_pred = model.predict(x=[x_cnn1d_test, x_lstm_test])

print(y_pred.shape, y_test.shape)
ind = 0

# plotear multiples resultados
plot_multiple_xy_results(y_pred, y_test, target_col, ind,
                         folder_name="Environment")

model.save("models/flotation_environment.h5")

charge_model = tf.keras.models.load_model(
    "models/flotation_environment.h5", compile=False)
# charge_pred = charge_model.predict(x=[x_nn_test, x_lstm_test, x_cnn1d_test])
