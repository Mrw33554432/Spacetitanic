#!/usr/bin/env python
# coding: utf-8

# this file is converted from SpaceTitanicV0, and may contain error. We recommend to run the original file in Jupyter
# Notebook

# In[76]:


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# In[77]:


training_data = pd.read_csv("data/train.csv")


# In[78]:


training_data.fillna(0, inplace=True)


# In[79]:


training_data = training_data.to_numpy()


# In[80]:


training_data[:, 3][0].split('/')


# In[81]:


def split_3rd(x):
    if x == 0:
        return [0, 0, 0]
    else:
        a, b, c = x.split('/')
        return a, int(b), c


new_seperated_column = np.hstack(map(split_3rd, training_data[:, 3])).reshape(training_data.shape[0], 3)


# In[82]:


training_data = np.delete(training_data, 3, axis=1)


# In[83]:


training_data = np.hstack((training_data, new_seperated_column))


# In[84]:


def class_mapping(column_number):
    classes = {}
    index = 0
    for id in training_data[:, column_number]:
        if id not in classes:
            classes[id] = index
            index += 1
    return classes


# In[85]:


map_planet = class_mapping(1)


# In[86]:


training_data[:, 1] = np.vectorize(map_planet.get)(training_data[:, 1])


# In[87]:


def replace_with_mapped_value(column_number, map_name, data_set):
    data_set[:, column_number] = np.vectorize(map_name.get)(data_set[:, column_number])


# In[88]:


TF_map = class_mapping(2)


# In[89]:


destination_map = class_mapping(3)


# In[90]:


cabin0 = class_mapping(13)


# In[91]:


cabin2 = class_mapping(15)


# In[92]:


replace_with_mapped_value(2, TF_map, training_data)
replace_with_mapped_value(3, destination_map, training_data)
replace_with_mapped_value(5, TF_map, training_data)
replace_with_mapped_value(13, cabin0, training_data)
replace_with_mapped_value(15, cabin2, training_data)


# In[93]:




# In[94]:


np.random.shuffle(training_data)
training_data, test_data = training_data[:int(len(training_data) * 0.8)], training_data[int(len(training_data) * 0.8):]


# x=np.delete(training_data,[0,1,2],1).astype(float)
# y=training_data[:,[1]].astype(float)
# tensor_x=torch.Tensor(x)
# tensor_y=torch.Tensor(y)
# training_data=TensorDataset(tensor_x,tensor_y)
def to_tensor(np_array, removed_columns, target_column):
    removed_columns.append(target_column)
    x = np.delete(np_array, removed_columns, axis=1).astype(float)
    y = np_array[:, target_column].astype(int)
    y_with_zeros = []
    for class_element in y:
        y_with_zeros.append(np.zeros(len(TF_map)))
        y_with_zeros[-1][TF_map[class_element]] = 1
    y = y_with_zeros
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)
    tensor = TensorDataset(tensor_x, tensor_y)
    return tensor


# In[95]:


batch_size = 128

# Create data loaders.
train_dataloader = DataLoader(to_tensor(training_data, [0, 11], 12), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(to_tensor(test_data, [0, 11], 12), batch_size=batch_size)

for X, y in train_dataloader:
    print(X.shape)
    print(y.shape, y.dtype)
    break

# # Display sample data
# figure = plt.figure(figsize=(10, 8))
# cols, rows = 5, 5
# for i in range(1, cols * rows + 1):
#     idx = torch.randint(len(test_data), size=(1,)).item()
#     img, label = test_data[idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


# In[96]:


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
layer_size = 512
input_size = 13


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, len(TF_map)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)


# In[97]:


# loss_fn = nn.L1Loss()
loss_fn = nn.MSELoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[98]:


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # y=y.type(torch.float)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 8 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# In[99]:


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(pred.argmax(1).shape,y.shape)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    try:
        if correct>0.81:
            torch.save(model,'model/m3')
        elif correct>0.805:
            torch.save(model,'model/m2')
        elif correct>0.80:
            torch.save(model,'model/m1')
            print('save success')
    except Exception:
        pass


    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[100]:


epochs = 1000
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")


# In[ ]:


torch.save(model, 'model/m0')


# In[ ]:


final_test_data = pd.read_csv("data/test.csv")
final_test_data.fillna(0, inplace=True)
final_test_data = final_test_data.to_numpy()
new_seperated_column = np.hstack(map(split_3rd, final_test_data[:, 3])).reshape(final_test_data.shape[0], 3)
final_test_data = np.delete(final_test_data, 3, axis=1)
final_test_data = np.hstack((final_test_data, new_seperated_column))


# In[ ]:


replace_with_mapped_value(1, map_planet, final_test_data)
replace_with_mapped_value(2, TF_map, final_test_data)
replace_with_mapped_value(3, destination_map, final_test_data)
replace_with_mapped_value(5, TF_map, final_test_data)
replace_with_mapped_value(12, cabin0, final_test_data)
replace_with_mapped_value(14, cabin2, final_test_data)
final_test_data = np.delete(final_test_data, 11, axis=1)


# In[ ]:




# In[ ]:


model = torch.load('model/m2')


# In[ ]:


dct = {v: k for k, v in TF_map.items()}


# In[ ]:


result = []
for line in final_test_data:
    current_line = torch.Tensor(line[1:].astype(float)).to(device)
    pred = model(current_line)
    result.append([line[0], dct[pred.argmax().item()]])


# In[ ]:


result = [['PassengerId', 'Transported']] + result


# In[ ]:


np.savetxt("result.csv", result, delimiter=",", fmt='%s')


# In[ ]:




