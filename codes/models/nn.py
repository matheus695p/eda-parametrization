import torch
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.config.nn_config import arguments_parser
from src.neuralnet.reg_nn import RegNeuralNet
from src.neuralnet.datasets import Dataset
from src.utils.visualizations import plot_xy_results
from src.neuralnet.early_stopping import EarlyStopping
warnings.filterwarnings("ignore")

# cargar argumentos de la red
args = arguments_parser()
# gpu o cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "para hacer entrenamiento")
# read data
path_cleaned = "data/cleaned-data.csv"
df = pd.read_csv(path_cleaned)
df.drop(columns=["fecha"], inplace=True)

# target column
target = ['%cu_conc_final']

# separacion target features
columns = list(df.columns)
for col in target:
    columns.remove(col)

# target y features
y = df[target]
x = df[columns]

# dividir en train + val y test
x_trainval, x_test, y_trainval, y_test = train_test_split(
    x, y, test_size=args.validation_size,
    random_state=args.random_state)
# dividir train y val
x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval, test_size=0.1,
    random_state=args.random_state)

# data normalization
sc = MinMaxScaler(feature_range=(0, 1))
# training
x_train = sc.fit_transform(x_train)
# validación
x_val = sc.transform(x_val)
# testing
x_test = sc.transform(x_test)

# pasar a array.numpy
x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)
# determinar el número de features
num_features = x_train.shape[1]

# torch datasets
train_dataset = Dataset(
    torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
val_dataset = Dataset(
    torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
test_dataset = Dataset(
    torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

# lista de tensores a computar
target_list = []
for _, counter in train_dataset:
    target_list.append(counter)
print(torch.tensor(target_list).size())
target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]


# torch datasets
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

# modelo
model = RegNeuralNet(num_feature=num_features,
                     num_targets=1)
# mandar el modelo a la GPU
model.to(device)
print(model)

# función de costos
criterion = torch.nn.MSELoss()

# optimizador
optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)

# scheduler de la reducción de la tasa de aprendizaje
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min",
    factor=args.lr_factor,
    patience=args.lr_patience,
    verbose=True)
# setiar el early stopping de la red
early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                               delta=args.min_delta,
                               path='models/checkpoint.pt')

# listas de metricas de evaluación
loss_stats = {
    'train': [],
    "val": []}

print("Empezemos el entrenamiento ...")
for e in range(1, args.epochs+1):

    # Entrenamiento
    train_epoch_loss = 0
    train_epoch_acc = 0
    # llamar train de torch.nn
    model.train()
    # Datos de entrenamiento
    for x_train_batch, y_train_batch in train_loader:
        # batches de data
        x_train_batch, y_train_batch = x_train_batch.to(
            device), y_train_batch.to(device)
        # setiar a cero los gradientes [en pytorch son acumulativos]
        optimizer.zero_grad()
        # fordward hacia adelante
        y_train_pred = model(x_train_batch)
        # error de entrenamiento
        train_loss = criterion(y_train_pred.float(), y_train_batch.float())
        # backpropragation de la perdida
        train_loss.backward()
        optimizer.step()
        # loss
        train_epoch_loss += train_loss.item()

    # Validación [torch.no_grad() no hacer backpropagation]
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch, y_val_batch = x_val_batch.to(
                device), y_val_batch.to(device)
            y_val_pred = model(x_val_batch)
            val_loss = criterion(y_val_pred, y_val_batch)
            val_epoch_loss += val_loss.item()

    # guardar en el diccionario
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))

    # computo de loss promedio por epoca
    train_loss = train_epoch_loss / len(train_loader)
    val_loss = val_epoch_loss / len(val_loader)

    # learning rate bajada
    scheduler.step(val_loss)

    # early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        break

    print(f"Epoca {e+0: 03}: | Train Loss: {train_loss: .5f}",
          f"| Val Loss: {val_loss:.5f}")

# testear los resultados
y_pred_list = []
with torch.no_grad():
    model.eval()
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_test_pred = model(x_batch)
        print(y_test_pred.size())
        _, y_pred_tags = torch.max(y_test_pred, dim=1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
predictions = [a.squeeze().tolist() for a in y_pred_list]

predictions = np.reshape(predictions, (-1, 1))

plot_xy_results(predictions, y_test, index=str(1), name=target[0],
                folder_name="random-forest")
