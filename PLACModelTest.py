import torch
import torch.nn as nn
import torchaudio
import pyaudio
import numpy as np

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

# Parameters
input_size = 1
hidden_size = 128
num_layers = 2
sampling_rate = 44100  # Standard audio sampling rate
packet_loss_rate = 0.1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = PLCModel(input_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load('plc_model.pth'))
model.eval()

# Function to simulate packet loss and conceal it
def conceal_packet_loss(audio):
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    mask = (torch.rand(audio_tensor.shape) > packet_loss_rate).float().to(device)
    corrupted_audio = audio_tensor * mask

    with torch.no_grad():
        concealed_audio = model(corrupted_audio)
    
    return concealed_audio.squeeze().cpu().numpy()

# PyAudio setup
p = pyaudio.PyAudio()

# Define callback function for audio processing
def callback(indata, frame_count, time_info, status):
    if status:
        print(status)
    audio_data = np.frombuffer(indata, dtype=np.float32)
    concealed_audio = conceal_packet_loss(audio_data)
    return (concealed_audio.astype(np.float32).tobytes(), pyaudio.paContinue)

# Open input stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sampling_rate,
                input=True,
                output=True,
                frames_per_buffer=1024,
                stream_callback=callback)

print("Starting stream...")

# Start the stream
stream.start_stream()

# Keep the stream active for 10 seconds
import time
time.sleep(10)

# Stop the stream
stream.stop_stream()
stream.close()
p.terminate()

print("Stream closed.")
