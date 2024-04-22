import numpy as np
import soundfile as sf
from scipy.signal import wiener

def apply_wiener_filter(audio_path, output_path):
    # Load audio file
    data, samplerate = sf.read(audio_path)
    
    # Apply Wiener filter
    filtered_data = wiener(data)
    
    # Save filtered audio
    sf.write(output_path, filtered_data, samplerate)
    
    print("Filtered audio saved to:", output_path)

# Example usage
apply_wiener_filter("../sample-data/sm3_n2H.wav", "trgt.wav")
