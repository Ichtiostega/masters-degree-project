import librosa as lr
from librosa import display as ld
from matplotlib import pyplot as plt

x, sr = lr.load('Wilhelm_Scream.ogg', sr=44100)
plt.figure(figsize=(18,5))
plt.margins(x=0.1, y=0.1)

plt.subplot(1,3,1)
plt.title("Wave")
ld.waveplot(x, sr=sr)

plt.subplot(1,3,2)
X = lr.stft(x)
Xdb = lr.amplitude_to_db(abs(X))
plt.title("Spectrogram")
ld.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

plt.subplot(1,3,3)
chromagram = lr.feature.chroma_stft(x, sr=sr, hop_length=5)
plt.title("Chromogram")
ld.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=5, cmap='coolwarm')

plt.subplots_adjust(top=0.94, left=0.04, right=0.98)
plt.savefig('representations.png')

x, sr = lr.load('IDMT_sample.ogg')

chromagram = lr.feature.chroma_stft(x, sr=sr, hop_length=5)
plt.title("Chromogram")
ld.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=5, cmap='coolwarm')
plt.show()

X = lr.stft(x)
Xdb = lr.amplitude_to_db(abs(X))
plt.title("Spectrogram")
ld.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.show()
