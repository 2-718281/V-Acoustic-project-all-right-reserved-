import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt


fs, data = wavfile.read('HYD1.wav')


segment_duration = 1  # 1s
segment_length = segment_duration * fs  


num_segments = len(data) // segment_length  

for i in range(num_segments):
    start = i * segment_length
    end = (i + 1) * segment_length
    segment_data = data[start:end]
    
    # STFT
    f, t, Zxx = stft(segment_data, fs=fs)
    
  
    plt.figure(figsize=(10, 6))  
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title(f'Spectrogram of Segment {i + 1}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    

    plt.savefig(f'segment_{i+1}.png', dpi=300) 
    plt.close()  

# if less than desired time
if len(data) % segment_length != 0:
    remaining_data = data[num_segments * segment_length:]
    f, t, Zxx = stft(remaining_data, fs=fs)
    
    
    plt.figure(figsize=(10, 6)) 
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('Remaining Segment')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    
    # save
    plt.savefig('remaining_segment.png', dpi=300)  
    plt.close()  