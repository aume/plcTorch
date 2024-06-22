import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

import numpy as np

import wave
import sys
import pyaudio

# Hyper-parameters 

hidden_size = 128 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = 1024 
sequence_length = 1024
num_layers = 2

# audio sample params
frame_chunk = 1024

# load audio data
#
class AudioDataset(Dataset):
    def __init__(self, path, chunk_size):
        # init data
        audio_data = []
        with wave.open(path, 'rb') as wf:
            while len(data := wf.readframes(frame_chunk)):  # Requires Python 3.8+ for :=
                data_array = np.frombuffer(data, dtype='int16')
                if data_array.size < frame_chunk:
                    # filter dataset for smaller chunks
                    continue
                data_array = torch.as_tensor(data_array)
                data_array = data_array.to(torch.float32)
                audio_data.append(torch.as_tensor(data_array))
                print(data_array)
        self.x_data = audio_data[:len(audio_data)-1]
        self.y_data = audio_data[1:len(audio_data)]
        self.n_samples = len(self.x_data)

    def get_data(self):
        return self.x_data, self.y_data
    
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


class MyModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 200) # input_size, output_size
        self.activation = nn.ReLU()
        # self.rnn = torch.nn.LSTM(1024, hidden_dim) # input_size, hidden_size, num_layers 
        self.linear2 = nn.Linear(200, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


    
audio_dataset = AudioDataset(sys.argv[1], frame_chunk)
training_loader = DataLoader(audio_dataset, batch_size=12,
                        shuffle=False, num_workers=0)
validation_dataset = AudioDataset(sys.argv[2], frame_chunk)
validation_loader = DataLoader(audio_dataset, batch_size=12,
                        shuffle=False, num_workers=0)

model = MyModel(frame_chunk, frame_chunk)

# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    #inputs, labels = dataset.get_data()
    #print(len(labels))
    #for i in range(0,len(inputs)):#, labels in dataset.get_data():
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # input = inputs[i]
        # label = labels[i]
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # output = model(input)
        outputs = model(inputs)
        # Compute the loss and its gradients
        
        # if len(output) != len(label):
        #     continue

        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

# x = torch.randn(64, frame_chunk)
# print(model(x).shape())

# # with tempfile.TemporaryDirectory() as tempdir:
# #     path = f"{tempdir}/save_example_default.wav"
# #     torchaudio.save(path, waveform, sample_rate)
# #     inspect_file(path)


# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# # Train the model
# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):  
#         # origin shape: [100, 1, 28, 28]
#         # resized: [100, 784]
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()

#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')