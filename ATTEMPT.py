import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
from skimage.morphology import binary_erosion, square
import matplotlib.pyplot as plt

# 1. 加载高采样率的音频数据
sample_rate, audio_data = wavfile.read('HYD1.wav')

# 如果音频数据是立体声，选择一个声道
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0]

# 2. 定义每个块的长度，按秒分段处理
segment_duration = 1  # 1秒
segment_length = segment_duration * sample_rate  # 对应的样本数

# 保存处理后的时域信号
processed_audio = []

# 3. 逐块处理并进行逆STFT
num_segments = len(audio_data) // segment_length

for i in range(num_segments):
    # 取得当前块的数据
    start = i * segment_length
    end = (i + 1) * segment_length
    segment_data = audio_data[start:end]
    
    # 对该段数据进行STFT
    f, t, Zxx = stft(segment_data, fs=sample_rate)
    
    # 二值化处理
    magnitude = np.abs(Zxx)
    threshold = np.median(magnitude)  # 使用中值作为阈值
    binary_magnitude = np.where(magnitude > threshold, 1, 0)
    
    # 腐蚀操作
    eroded_magnitude = binary_erosion(binary_magnitude, footprint=square(3)) * magnitude  # 腐蚀后的振幅
    
    # 将振幅与原始相位信息结合
    Zxx_processed = eroded_magnitude * np.exp(1j * np.angle(Zxx))
    
    # 逆短时傅里叶变换 (iSTFT)
    _, reconstructed_segment = istft(Zxx_processed, fs=sample_rate)
    
    # 保存处理后的时域信号段
    processed_audio.extend(reconstructed_segment)
    
    # 绘制处理后的时频图
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx_processed), shading='gouraud')
    plt.title(f'Processed Spectrogram of Segment {i + 1}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    
    # 保存每段处理后的时频谱图为PNG文件
    plt.savefig(f'processed_spectrogram_segment_{i+1}.png', dpi=300)
    plt.close()

# 4. 处理剩余部分
if len(audio_data) % segment_length != 0:
    remaining_data = audio_data[num_segments * segment_length:]
    f, t, Zxx = stft(remaining_data, fs=sample_rate)
    
    # 二值化处理
    magnitude = np.abs(Zxx)
    threshold = np.median(magnitude)
    binary_magnitude = np.where(magnitude > threshold, 1, 0)
    
    # 腐蚀操作
    eroded_magnitude = binary_erosion(binary_magnitude, footprint=square(3)) * magnitude
    
    # 将振幅与原始相位结合
    Zxx_processed = eroded_magnitude * np.exp(1j * np.angle(Zxx))
    
    # 逆STFT
    _, reconstructed_segment = istft(Zxx_processed, fs=sample_rate)
    
    # 保存剩余部分的处理结果
    processed_audio.extend(reconstructed_segment)
    
    # 绘制处理后的时频图
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx_processed), shading='gouraud')
    plt.title('Processed Spectrogram of Remaining Segment')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    
    # 保存剩余部分的时频谱图
    plt.savefig('processed_spectrogram_remaining_segment.png', dpi=300)
    plt.close()

# 5. 将处理后的时域信号保存为新的WAV文件
processed_audio = np.array(processed_audio, dtype=np.int16)
wavfile.write('processed_HYD1.wav', sample_rate, processed_audio)

# 6. 可视化处理后的音频信号
plt.figure(figsize=(10, 4))
plt.plot(processed_audio)
plt.title('Processed Audio Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
