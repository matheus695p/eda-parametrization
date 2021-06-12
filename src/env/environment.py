import numpy as np
import pandas as pd


class FlotationEnv:
    """
    Crear el entorno de entrenamiento del proceso de flotación de la minera
    construyendo los principales del entorno que son:
        __get_data, __get_window, __get
    """

    def __init__(self, timesteps, df_name,
                 target_cols=['%cu_conc_final', '%cu_cola_final']):
        """
        Constructor del enviroment del agente
        Parameters
        ----------
        window_size : int
            tamaño de ventana temporal hacia atrás en la data.
        stock_name : string
            nombre del archivo.
        """
        # columnas target
        self.target_cols = target_cols
        # traer el archivo como pandas dataframe
        self.data = self.__get_data(df_name)

        # definir el estado en cúal estamos
        self.states = self.__get_states(self.data, timesteps)
        # indice último
        self.index = -1
        # traer esa última data
        self.last_data_index = len(self.data) - 1

    def __get_data(self, key):
        """
        Leer el dataframe y devolverlo como un dataframe
        Parameters
        ----------
        key : string
            nombre del archivo.
        Returns
        -------
        vec : list
            precio de cerrada del activo.
        """
        data = pd.read_csv("data/" + key + ".csv", index=False)
        return data

    def __get_actual_state(self, data, window_size):
        """
        Sacar la matriz de estados dado un indice del dataframe
        Parameters
        ----------
        data : dataframe
            dataframe con el entorno.
        window_size : int
            tamaño de la ventana temporal.

        Returns
        -------
        states: retorna la matriz de los estados pasados.

        """
        states = data.iloc[self.index-window_size:self.index, :].to_numpy()
        states = states.reshape(-1, states.shape[0], states.shape[1])
        return states

    def reset(self):
        """
        Resetear el últiumo indice, dado que no hay más para donde avanzar
        """
        self.index = -1
        return self.states[0], self.data[0]

    def get_next_state_reward(self, action, bought_price=None):
        """
        Obtener la recompenza del estado siguiente, la idea del siguiente
        estado es que pueda avanzar por el vector de precios, como el
        siguiente estado
        Parameters
        ----------
        action : int
            vender, quedarse o comprar.
        bought_price : TYPE, optional
            DESCRIPTION. The default is None.
        Returns
        -------
        next_state : TYPE
            DESCRIPTION.
        next_price_data : TYPE
            DESCRIPTION.
        reward : TYPE
            DESCRIPTION.
        done : TYPE
            DESCRIPTION.
        """
        # avanzar un estado en el problema
        self.index += 1
        # cuando llegue al final, que se setee a cero
        if self.index > self.last_data_index:
            self.index = 0
        # el estado siguiente es sumar uno al indice
        next_state = self.states[self.index + 1]

        # el precio siguiente es el mismo
        next_price_data = self.data[self.index + 1]
        price_data = self.data[self.index]

        # recompenza a cero
        reward = 0
        # si la acción es 2 y el precio de compra no es el default
        if action == 2 and bought_price is not None:
            reward = max(price_data - bought_price, 0)

        # solo termina el episodio cuando paso todo el vector
        done = True if self.index == self.last_data_index - 1 else False
        return next_state, next_price_data, reward, done


def lstm_preparation(array, timesteps=5):
    """
    Preparar los datos para la predicción con la lstm
    Parameters
    ----------
    array : numpy.array
        array.
    timesteps : int, optional
        cantidad de tiemsteps que se harán las predicciones.
        The default is 5.
    Returns
    -------
    x_train : array
        matriz de entrenamiento de las celdas lstm.
    y_train : array
        salida de las celdas.
    """
    x_train = []
    y_train = []
    for i in range(timesteps, array.shape[0]):
        x_train.append(array[i-timesteps:i])
        y_train.append(array[i][0:array.shape[1]])
    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    return x_train, y_train


def nn_preparation(array, names, target_col, timesteps=5):
    """
    Hacer la preparación de la red neuronal
    Parameters
    ----------
    array : numpy.array
        array con todas las variables.
    names : list
        nombre de todas las columnas.
    target_col : string or list
        nombre de la/as columna/as target/s.
    timesteps : int, optional
        cantidad de tiemsteps que se harán las predicciones.
        The default is 5.
    Returns
    -------
    x : array
        x en numpy.
    y : array
        target en numpy.
    """
    df = pd.DataFrame(array, columns=names)
    df = df.iloc[timesteps:, :]
    df.reset_index(drop=True, inplace=True)

    if len(target_col) == 1:
        y = df[[target_col]]
    else:
        y = df[target_col]

    x = df.drop(columns=target_col)
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y
