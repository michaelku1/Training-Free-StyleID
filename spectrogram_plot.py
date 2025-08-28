from utils.plot_spectrogram import plot_spectrogram, plot_single_spectrogram

# EGDB and sinsy vocal
# style_path = "/mnt/gestalt/home/mku666/wah_emulation/EGDB_DI/233.wav"
# content_path = "/mnt/gestalt/home/mku666/vocal2guitar/vocals/233_sinsy.wav"
# output_path = "/home/mku666/riffusion-hobby/output.wav" 

# musicTI dataset

# audio fx dataset
# style_path = "/mnt/gestalt/home/mku666/wah_emulation/NA_Wah_75/NA_WahFilter_7.5_Power_True_Bypass_False/223.wav"
# content_path = "/mnt/gestalt/home/mku666/wah_emulation/EGDB_DI/233.wav"
# output_path = "/home/mku666/riffusion-hobby/results/wah_wah_effect_test.wav"
# output_path = "/home/mku666/riffusion-hobby/results/wah_wah_effect_test_t_step_5.wav"

# Use a two-second segment starting from 0 seconds
# start_time = 0  # Start from the beginning
# duration = 10  # Two seconds duration

# plot_spectrogram(style_path, content_path, save_path='spectrogram.png', name1='Style Spectrogram', name2='Content Spectrogram', start_time=start_time, duration=duration)
# plot_single_spectrogram(audio_path, save_path='output_spectrogram.png', title='Output Spectrogram', start_time=start_time, duration=duration)


if __name__ == "__main__":
    import argparse

    start_time = 0  # Start from the beginning
    duration = 10  # Two seconds duration

    parser = argparse.ArgumentParser(description='Plot spectrograms')
    # parser.add_argument('--style_path', type=str, required=True, help='Path to the style audio file')
    # parser.add_argument('--content_path', type=str, required=True, help='Path to the content audio file')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output audio file')
    args = parser.parse_args()

    # plot_spectrogram(args.style_path, args.content_path, args.output_path, save_path='spectrogram.png', name1='Style Spectrogram', name2='Content Spectrogram', name3='Output Spectrogram', start_time=start_time, duration=duration)
    plot_single_spectrogram(args.audio_path, save_path=args.output_path, title='Output Spectrogram', start_time=start_time, duration=duration)

