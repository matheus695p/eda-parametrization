import keras.backend as K


def handler_loss_function(factor=1.5):
    """
    Función de costos para balancear aprendizaje de los pesos de la red
    dado que prefiere una clase mejor que la otra

    Parameters
    ----------
    factor : float, optional
        factor por el cual multiplicar la clase de
        % de cu en cola final. The default is 1.5.

    Returns
    -------
    flotation_loss_function
        DESCRIPTION.

    """
    # Retorna la función de costos de cosmos con penalización
    def flotation_loss_function(y_true, y_pred):
        """
        Función de costos implementación, el error cuadratico medio
        pero tiene factores multiplicativos en la clase perjudicada por el
        rango de su variable target

        Parameters
        ----------
        y_true : array
            observaciones.
        y_pred : TYPE
            predicciones.

        Returns
        -------
        loss : function
            función de costos.

        """
        # Covertir a tensor de tensorflow con keras como backend
        y_true = K.cast(y_true, dtype='float32')
        y_pred = K.cast(y_pred, dtype='float32')

        mean = K.mean(y_true, axis=0)
        ratio = mean[0] / mean[1]

        # Reshape como vector
        y_true = K.reshape(y_true, (-1, 2))
        y_pred = K.reshape(y_pred, (-1, 2))

        size_train = K.shape(y_pred)[0]
        size_train = K.reshape(size_train, (-1, 1))
        loss = K.square(y_pred - y_true)
        loss = loss[:, 0] + loss[:, 1] * ratio * factor
        loss = K.mean(loss)
        return loss

    return flotation_loss_function
