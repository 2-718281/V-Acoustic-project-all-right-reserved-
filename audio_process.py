import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import emd


over = 0.5
fs = 250000


fs1, data1 = scipy.io.wavfile.read('HYD1.wav')
fs2, data2 = scipy.io.wavfile.read('HYD2.wav')
fs3, data3 = scipy.io.wavfile.read('HYD3.wav')

time1 = np.linspace(0,len(data1)/fs, fs)
time2 = np.linspace(0,len(data2)/fs, fs)
time3 = np.linspace(0,len(data3)/fs, fs)


# correlation 
corr12 = scipy.signal.correlate(data1,data2)
corr23 = scipy.signal.correlate(data2,data3)  

plt.plot(corr12)
plt.plot(corr23)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

plt.show()


#emd approach
'''
imf = emd.sift.sift(data2)
print(imf.shape)

IP, IF, IA = emd.spectra.frequency_transform(imf, fs, 'hilbert')
# Define frequency range (low_freq, high_freq, nsteps, spacing)
freq_range = (0.1, 10, 80, 'log')
f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
emd.plotting.plot_imfs(imf)

fig = plt.figure(figsize=(10, 6))
emd.plotting.plot_hilberthuang(hht, time2, f,
                               time_lims=(2, 4), freq_lims=(0.1, 15),
                               fig=fig, log_y=True)
'''


#looking for the peak 
'''
fft1 = scipy.fft.fft(data1)

def normal(data):
    normalised  = data/np.max(np.abs(data))
    return normalised

normalised1 = normal(data1)
normalised2 = normal(data2)
normalised3 = normal(data3)

peak1,_ = scipy.signal.find_peaks(normalised1,height = over)
peak2,_ = scipy.signal.find_peaks(normalised2,height = over)
peak3,_ = scipy.signal.find_peaks(normalised3,height = over)

time1 = peak1/fs1
time2 = peak2/fs2
time3 = peak3/fs3

time1_before_4_5 = time1[time1 <= 4.5]
time2_before_4_5 = time1[time1 <= 4.5]
time3_before_4_5 = time1[time1 <= 4.5]

def peak_f_seg_1(peaks,time,normal):
    time_differences = np.diff(time)
    for i, time_diff in enumerate(time_differences):
        print(f"Peak {i+1} to Peak {i+2}: {time_diff:.6f} seconds")
    if len(time_differences) > 0:
        average_time_diff = np.mean(time_differences)
        print(f"avg peak before 4.5s is: {average_time_diff:.6f} 秒")
    else:
        print("XD")    

def peak_f_seg_2(peaks,time,normal):
    for i in range(len(peaks)):
        print(f"Peak at {time[i]} seconds with amplitude {normal[peaks[i]]}")

def peak_diff(time):
    timediff = np.diff(time)
    timeavediff = np.average(timediff)
    return timeavediff



peak_f_seg_1(peak1,time1_before_4_5,normalised1)
peak_f_seg_1(peak2,time2_before_4_5,normalised2)
peak_f_seg_1(peak3,time3_before_4_5,normalised3)

peak_diff(time1_before_4_5)
print(peak_diff(time2_before_4_5))
peak_diff(time3_before_4_5)




plt.plot(np.arange(len(normalised2)) / fs2, normalised2)
#plt.plot(np.arange(len(fft1)),abs(fft1))
plt.plot(time2, normalised2[peak2], "o")  # 标记峰值
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Waveform and Detected Peaks')
plt.show()
'''