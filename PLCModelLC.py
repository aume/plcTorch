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
packet_loss_rate = 0.1  # This can be ignored for real-time detection

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = PLCModel(input_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load('plc_model.pth'))
model.eval()

# Function to simulate packet loss and conceal it
def conceal_packet_loss(audio, last_valid_packet):
    audio_tensor = torch.tensor(audio, dtype=torch.float16).unsqueeze(0).unsqueeze(-1).to(device)
    corrupted_audio = audio_tensor.clone()
    if last_valid_packet is not None:
        corrupted_audio[corrupted_audio == 0] = last_valid_packet[corrupted_audio == 0]

    with torch.no_grad():
        concealed_audio = model(corrupted_audio)
    
    return concealed_audio.squeeze().cpu().numpy()

# PyAudio setup
p = pyaudio.PyAudio()

# Initialize variables to keep track of packet states
last_valid_packet = None

# Define callback function for audio processing
def callback(indata, frame_count, time_info, status):
    global last_valid_packet
    if status:
        print(status)

    audio_data = np.frombuffer(indata, dtype=np.float16)
    
    # Detect packet loss: assuming zero data indicates a corrupt packet
    if np.all(audio_data == 0):
        print("Corrupt packet detected!")
        if last_valid_packet is not None:
            audio_data = conceal_packet_loss(audio_data, last_valid_packet)
    else:
        last_valid_packet = torch.tensor(audio_data, dtype=torch.float16).unsqueeze(0).unsqueeze(-1).to(device)

    return (audio_data.astype(np.float16).tobytes(), pyaudio.paContinue)

# Open input stream
stream = p.open(format=pyaudio.paFloat16,
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
