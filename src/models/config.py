import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore", category=DeprecationWarning)


def arguments_parser():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    una red deep renewal
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="haciendo weon a python", default="1")
    # agregar donde correr y guardar datos
    # nombre
    parser.add_argument('--env_name', type=str,
                        default="HalfCheetahBulletEnv-v0")
    # semilla
    parser.add_argument('--seed', type=int, default=0)
    # Número de of iteraciones/timesteps durante las cuales el modelo elige una
    # acción al azar, y después de las cuales comienza a usar la red de
    # políticas
    parser.add_argument('--start_timesteps', type=int, default=1e4)
    # Con qué frecuencia se realiza el paso de evaluación
    # (después de cuántos pasos timesteps)
    parser.add_argument('--eval_freq', type=int, default=5e3)
    # Número total de iteraciones/timesteps
    parser.add_argument('--max_timesteps', type=int, default=5e5)
    # Check Boolean para saber si guardar o no el modelo pre-entrenado
    parser.add_argument('--save_models', type=bool, default=True)
    # Ruido de exploración: desviación estándar del ruido de exploración
    # gaussiano
    parser.add_argument('--expl_noise', type=float, default=0.1)
    # Tamaño del bloque
    parser.add_argument('--batch_size', type=int, default=100)
    # Factor de descuento gamma, utilizado en el cáclulo de la recompensa de
    # descuento total
    parser.add_argument('--discount', type=float, default=0.99)
    # Ratio de actualización de la red de objetivos
    parser.add_argument('--tau', type=float, default=0.005)
    # Desviación estándar del ruido gaussiano añadido a las acciones
    # para fines de exploración
    parser.add_argument('--policy_noise', type=float, default=0.2)
    # Valor máximo de ruido gaussiano añadido a las acciones (política)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    # Número de iteraciones a esperar antes de actualizar la red de políticas
    # (actor modelo)
    parser.add_argument('--policy_freq', type=int, default=2)

    args = parser.parse_args()
    return args
