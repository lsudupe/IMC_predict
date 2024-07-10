
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

# Ruta al archivo pickle
file_path = '/Users/medinils/Desktop/IMC_Spatial_predictions/graph/my_dataset.pkl'

# Abrir el archivo en modo lectura binaria ('rb')
with open(file_path, 'rb') as f:
    loaded_dataset = pickle.load(f)


####definir modelo
##GCN
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Aplica un pooling global a los nodos
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

## GAT
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=1, concat=True, dropout=0.6)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Apply global mean pooling over the nodes in the batch
        x = self.fc(x)
        return F.log_softmax(x, dim=1)




####Paso 1: Dividir los Datos en Entrenamiento y Prueba
def split_dataset(dataset, train_ratio=0.8):
    # Determinar el punto de corte para el conjunto de entrenamiento
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    # Dividir el dataset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


# Aplicar la función de división
train_dataset, test_dataset = split_dataset(loaded_dataset)


# Asegúrate de usar esta función de colación al crear el DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


####Paso 2: Configurar el Entrenamiento y la Evaluación del Modelo
# Definir el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(input_dim=3, hidden_dim=64, output_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    start_time = time.time()
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    end_time = time.time()
    training_time = end_time - start_time
    return total_loss / len(train_loader), training_time

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
        y_true.extend(data.y.tolist())
        y_pred.extend(pred.tolist())
    accuracy = correct / total
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1


####Paso 3:Evaluación del Modelo
train_losses = []
test_accuracies = []

for epoch in range(10):
    train_loss, train_time = train()
    accuracy, precision, recall, f1 = evaluate(test_loader)
    train_losses.append(train_loss)
    train_times.append(train_time)
    test_accuracies.append(accuracy)
    test_precisions.append(precision)
    test_recalls.append(recall)
    test_f1s.append(f1)
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Time: {train_time:.2f}s, Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')


####Paso 4:Plots
epochs = range(1, 11)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(epochs, train_times, label='Training Time', marker='o')
plt.title('Training Time (seconds)')
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(epochs, test_accuracies, label='Accuracy', marker='o')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)