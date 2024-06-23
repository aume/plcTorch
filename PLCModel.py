import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os
from sklearn.model_selection import train_test_split

class AudioDataset(Dataset):
    def __init__(self, file_list, packet_loss_rate=0.1):
        self.file_list = file_list
        self.packet_loss_rate = packet_loss_rate

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio, sample_rate = torchaudio.load(self.file_list[idx])
        audio = audio.squeeze(0)  # Ensure it's mono
        # Simulate packet loss
        mask = (torch.rand(audio.shape) > self.packet_loss_rate).float()
        corrupted_audio = audio * mask
        return corrupted_audio, audio, mask

def collate_fn(batch):
    corrupted_audios, original_audios, masks = zip(*batch)
    corrupted_audios = pad_sequence(corrupted_audios, batch_first=True)
    original_audios = pad_sequence(original_audios, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)
    return corrupted_audios.unsqueeze(-1), original_audios.unsqueeze(-1), masks.unsqueeze(-1)

class PLCModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PLCModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for corrupted_audio, original_audio, mask in dataloader:
            corrupted_audio = corrupted_audio.to(device)
            original_audio = original_audio.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            outputs = model(corrupted_audio)
            loss = criterion(outputs * mask, original_audio * mask)
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for corrupted_audio, original_audio, mask in dataloader:
            corrupted_audio = corrupted_audio.to(device)
            original_audio = original_audio.to(device)
            mask = mask.to(device)

            outputs = model(corrupted_audio)
            loss = criterion(outputs * mask, original_audio * mask)
            total_loss += loss.item()

    print(f'Average Loss: {total_loss / len(dataloader):.4f}')

# Parameters
input_size = 1  # Mono audio
hidden_size = 128
num_layers = 2
num_epochs = 20
batch_size = 16
learning_rate = 0.001
packet_loss_rate = 0.1

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')

# Load data
file_list = [os.path.join('audio_data', f) for f in os.listdir('audio_data') if f.endswith('.wav')]
train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)
train_dataset = AudioDataset(train_files, packet_loss_rate)
test_dataset = AudioDataset(test_files, packet_loss_rate)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model, loss, and optimizer
model = PLCModel(input_size, hidden_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train and evaluate the model
train(model, train_loader, criterion, optimizer, num_epochs)
evaluate(model, test_loader)

# Save the model
torch.save(model.state_dict(), 'plc_model.pth')
