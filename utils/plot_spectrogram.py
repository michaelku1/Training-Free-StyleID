# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import spectrogram
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



def plot_single_spectrogram(audio_path, save_path='single_spectrogram.png', title='Spectrogram', start_time=0, duration=None):
    """
    Plot spectrogram of a single audio file

    Args:
        audio_path: path to audio file
        save_path: path to save the plot (default: 'single_spectrogram.png')
        title: title for the plot (default: 'Spectrogram')
        start_time: start time in seconds (default: 0)
        duration: duration in seconds, None for full audio (default: None)
    """

    # Load audio file
    audio, sr = librosa.load(audio_path, offset=start_time, duration=duration)

    # Compute spectrogram
    n_fft = 2048
    hop_length = 512
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # Convert to decibel scale
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot spectrogram
    img = librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spectrogram(x1, x2, x3, save_path='spectrogram', name1='Spectrogram Plot 1', name2='Spectrogram Plot 2', name3='Spectrogram Plot 3', start_time=0, duration=None):
    """
    Plot spectrogram of three audio files

    Args:
        x1: content path to first audio file
        x2: style path to second audio file
        x3: generated audio path to third audio file
        start_time: start time in seconds (default: 0)
        duration: duration in seconds, None for full audio (default: None)
    """

    # Load audio files with specified segment
    audio1, sr1 = librosa.load(x1, offset=start_time, duration=duration)
    audio2, sr2 = librosa.load(x2, offset=start_time, duration=duration)
    audio3, sr3 = librosa.load(x3, offset=start_time, duration=duration) 

    # Resample if necessary
    if sr1 != sr2:
        audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        sr = sr1
    else:
        sr = sr1

    # Compute spectrograms
    n_fft = 2048
    hop_length = 512
    spec1 = librosa.stft(audio1, n_fft=n_fft, hop_length=hop_length)
    spec2 = librosa.stft(audio2, n_fft=n_fft, hop_length=hop_length)
    spec3 = librosa.stft(audio3, n_fft=n_fft, hop_length=hop_length)

    # Convert to decibel scale
    spec1_db = librosa.amplitude_to_db(np.abs(spec1), ref=np.max)
    spec2_db = librosa.amplitude_to_db(np.abs(spec2), ref=np.max)
    spec3_db = librosa.amplitude_to_db(np.abs(spec3), ref=np.max)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot spectrogram 1
    img1 = librosa.display.specshow(spec1_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax1, cmap='viridis')
    ax1.set_title(name1)
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

    # Plot spectrogram 2
    img2 = librosa.display.specshow(spec2_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax2, cmap='viridis')
    ax2.set_title(name2)
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

    # Plot spectrogram 3
    img3 = librosa.display.specshow(spec3_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax3, cmap='viridis')
    ax3.set_title(name3)
    fig.colorbar(img3, ax=ax3, format='%+2.0f dB')

    # Plot difference spectrogram
    # spec_diff = spec1_db - spec2_db
    # img3 = librosa.display.specshow(spec_diff, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax3, cmap='coolwarm')
    # ax3.set_title('Difference (Spec1 - Spec2)')
    # fig.colorbar(img3, ax=ax3, format='%+2.0f dB')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
