import os
import fluidsynth
import numpy as np
import librosa
from scipy.io.wavfile import write
from tqdm import tqdm
import random
import glob
from scipy import signal
import json

# --- Configuration matching your training script ---
SOUNDFONT_DIR = 'sound_fonts'
NOISE_DIR = '03_Augmented_Audio/noise_sources'
OUTPUT_DIR = 'audio_files1'  # Matching your AUDIO_PATH
SAMPLE_RATE = 44100
DURATION_SECONDS = 2.0
SAMPLES_PER_FONT_ITEM = 30  # Increased for better coverage
SILENCE_SAMPLES_PER_NOISE_FILE = 50
SNR_DB_RANGE = (-10, 25)
VELOCITIES = [40, 60, 80, 90, 100, 110, 127]  # More velocity variations
MAX_WAV_VAL = 32767.0

# --- Enhanced Augmentation Probabilities ---
REVERB_PROBABILITY = 0.8
PITCH_SHIFT_PROBABILITY = 0.6
TIME_STRETCH_PROBABILITY = 0.5
COMPRESSION_PROBABILITY = 0.4
EQ_PROBABILITY = 0.5
PHONE_RECORDING_PROBABILITY = 0.3

# --- Note and Chord Definitions (matching your training) ---
NOTES = ['A0', 'A#0', 'B0']
for i in range(1, 8):
    NOTES.extend([f'C{i}', f'C#{i}', f'D{i}', f'D#{i}', f'E{i}', f'F{i}', 
                  f'F#{i}', f'G{i}', f'G#{i}', f'A{i}', f'A#{i}', f'B{i}'])
NOTES.append('C8')

# Enhanced chord types for better musical coverage
CHORDS = {
    'Major': (0, 4, 7), 'Minor': (0, 3, 7), 'Diminished': (0, 3, 6),
    'Augmented': (0, 4, 8), 'Suspended2': (0, 2, 7), 'Suspended4': (0, 5, 7),
    'Major7th': (0, 4, 7, 11), 'Minor7th': (0, 3, 7, 10), 'Dominant7th': (0, 4, 7, 10),
    'MinorMajor7th': (0, 3, 7, 11), 'Diminished7th': (0, 3, 6, 9),
    'Major6th': (0, 4, 7, 9), 'Minor6th': (0, 3, 7, 9),
    'Add9': (0, 4, 7, 14), 'MinorAdd9': (0, 3, 7, 14)
}

# --- Helper Functions ---
SHARP_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note_sharp(midi_num):
    octave = (midi_num // 12) - 1
    note_index = midi_num % 12
    return f"{SHARP_NOTES[note_index]}{octave}"

def simulate_phone_recording(audio, sr):
    """Simulate phone recording characteristics"""
    # Apply phone frequency response (bandpass filter)
    nyquist = sr // 2
    low = 300 / nyquist
    high = 3400 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    audio = signal.filtfilt(b, a, audio)
    
    # Add slight compression
    threshold = 0.3
    ratio = 4.0
    audio = np.where(np.abs(audio) > threshold,
                    np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
                    audio)
    
    # Add quantization noise (simulate ADC)
    bits = 12  # Phone ADC simulation
    audio = np.round(audio * (2**(bits-1))) / (2**(bits-1))
    
    return audio

def apply_dynamic_range_compression(audio, threshold=0.5, ratio=3.0):
    """Apply dynamic range compression"""
    compressed = np.where(np.abs(audio) > threshold,
                         np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
                         audio)
    return compressed

def apply_eq_augmentation(audio, sr):
    """Apply random EQ changes"""
    # Random bass boost/cut
    bass_gain = random.uniform(-6, 6)  # dB
    # Random treble boost/cut
    treble_gain = random.uniform(-6, 6)  # dB
    
    # Apply bass filter
    nyquist = sr // 2
    bass_freq = 200 / nyquist
    b_bass, a_bass = signal.butter(2, bass_freq, btype='low')
    bass_component = signal.filtfilt(b_bass, a_bass, audio)
    
    # Apply treble filter
    treble_freq = 2000 / nyquist
    b_treble, a_treble = signal.butter(2, treble_freq, btype='high')
    treble_component = signal.filtfilt(b_treble, a_treble, audio)
    
    # Apply gains
    bass_component *= (10 ** (bass_gain / 20))
    treble_component *= (10 ** (treble_gain / 20))
    
    # Combine
    mid_component = audio - bass_component - treble_component
    return bass_component + mid_component + treble_component

def mix_with_noise(clean_audio, noise_audio, snr_db):
    """Enhanced noise mixing with power normalization"""
    clean_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    
    if clean_power < 1e-10 or noise_power < 1e-10:
        return clean_audio
    
    snr = 10 ** (snr_db / 10)
    scale_factor = np.sqrt(clean_power / (snr * noise_power))
    
    # Ensure noise doesn't clip the result
    mixed = clean_audio + (noise_audio * scale_factor)
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed / max_val * 0.95
    
    return mixed

def save_audio(path, audio_data, sr):
    """Save audio with proper normalization"""
    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        normalized = audio_data / max_abs
        audio_int16 = np.int16(normalized * MAX_WAV_VAL)
        write(path, sr, audio_int16)

def apply_augmentations(audio, noise_files):
    """Apply comprehensive augmentations for real-world robustness"""
    augmented = audio.copy()
    
    # Phone recording simulation
    if random.random() < PHONE_RECORDING_PROBABILITY:
        augmented = simulate_phone_recording(augmented, SAMPLE_RATE)
    
    # Pitch shifting (smaller range for realism)
    if random.random() < PITCH_SHIFT_PROBABILITY:
        n_steps = random.uniform(-0.3, 0.3)
        augmented = librosa.effects.pitch_shift(y=augmented, sr=SAMPLE_RATE, n_steps=n_steps)

    # Time stretching (smaller range)
    if random.random() < TIME_STRETCH_PROBABILITY:
        rate = random.uniform(0.97, 1.03)
        augmented = librosa.effects.time_stretch(y=augmented, rate=rate)
    
    # Dynamic range compression
    if random.random() < COMPRESSION_PROBABILITY:
        threshold = random.uniform(0.3, 0.7)
        ratio = random.uniform(2.0, 6.0)
        augmented = apply_dynamic_range_compression(augmented, threshold, ratio)
    
    # EQ augmentation
    if random.random() < EQ_PROBABILITY:
        augmented = apply_eq_augmentation(augmented, SAMPLE_RATE)
    
    # Add noise (high probability for robustness)
    if noise_files and random.random() < 0.8:
        noise_path = random.choice(noise_files)
        try:
            noise_full, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)
            if len(noise_full) >= len(augmented):
                start = random.randint(0, len(noise_full) - len(augmented))
                noise_segment = noise_full[start:start + len(augmented)]
                snr_db = random.uniform(*SNR_DB_RANGE)
                augmented = mix_with_noise(augmented, noise_segment, snr_db)
        except Exception:
            pass
    
    return augmented

def generate_enhanced_dataset():
    """Generate dataset matching your training script structure"""
    soundfonts = glob.glob(os.path.join(SOUNDFONT_DIR, '*.sf2'))
    noise_files = glob.glob(os.path.join(NOISE_DIR, '*.*'))

    print(f"Found {len(soundfonts)} soundfonts and {len(noise_files)} noise files.")
    print("Generating enhanced dataset for real-world piano note recognition...")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for font_path in tqdm(soundfonts, desc="Processing Soundfonts"):
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        fs = fluidsynth.Synth(samplerate=SAMPLE_RATE)
        sfid = fs.sfload(font_path)
        fs.program_select(0, sfid, 0, 0)

        # Generate items: single notes and chords
        items_to_generate = NOTES + [(root, name, intervals) for root in NOTES for name, intervals in CHORDS.items()]

        for item in tqdm(items_to_generate, desc=f"Items ({font_name})", leave=False):
            note_numbers = []
            final_folder_name = ""
            
            # Handle single notes
            if isinstance(item, str):
                try:
                    note_midi = librosa.note_to_midi(item)
                    if note_midi < 21 or note_midi > 108:
                        continue
                    note_numbers = [note_midi]
                    final_folder_name = item.replace('#', 's')
                except:
                    continue
            
            # Handle chords
            else:
                root, chord_name, intervals = item
                try:
                    root_midi = librosa.note_to_midi(root)
                    note_numbers = [root_midi + i for i in intervals]
                    if any(n < 21 or n > 108 for n in note_numbers):
                        continue
                    
                    # Create folder name matching your training expectation
                    folder_name_parts = [midi_to_note_sharp(n).replace('#', 's') for n in note_numbers]
                    final_folder_name = f"{'_'.join(folder_name_parts)}_{chord_name}"
                except:
                    continue

            # Create output directory for this item
            output_dir = os.path.join(OUTPUT_DIR, final_folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate samples
            for i in range(SAMPLES_PER_FONT_ITEM):
                try:
                    velocity = random.choice(VELOCITIES)
                    
                    # Configure reverb
                    if random.random() < REVERB_PROBABILITY:
                        room = random.uniform(0.1, 1.0)
                        damp = random.uniform(0.1, 0.9)
                        width = random.uniform(0.1, 1.0)
                        level = random.uniform(0.1, 0.8)
                        fs.set_reverb(room, damp, width, level)
                    else:
                        fs.set_reverb(0.0, 0.0, 0.0, 0.0)

                    # Generate audio
                    for n in note_numbers:
                        fs.noteon(0, n, velocity)
                    
                    num_frames = int(SAMPLE_RATE * DURATION_SECONDS)
                    raw_samples = fs.get_samples(num_frames)
                    
                    for n in note_numbers:
                        fs.noteoff(0, n)
                    
                    # Convert to mono
                    stereo_samples = raw_samples.reshape(num_frames, 2)
                    mono_audio = librosa.to_mono(stereo_samples.astype(np.float32).T)
                    
                    # Apply augmentations
                    augmented_audio = apply_augmentations(mono_audio, noise_files)
                    
                    # Save with naming convention matching your training
                    filename = f"{final_folder_name}_{font_name}_{velocity}_{i+1}.wav"
                    filepath = os.path.join(output_dir, filename)
                    save_audio(filepath, augmented_audio, SAMPLE_RATE)
                    
                except Exception as e:
                    print(f"Error generating sample {i+1} for {final_folder_name}: {e}")
                    continue

        fs.delete()

    # Generate silence samples
    if noise_files:
        print("\nGenerating silence samples...")
        silence_dir = os.path.join(OUTPUT_DIR, 'silence')
        os.makedirs(silence_dir, exist_ok=True)
        required_len = int(SAMPLE_RATE * DURATION_SECONDS)
        
        for noise_path in tqdm(noise_files, desc="Silence Generation"):
            try:
                noise_full, _ = librosa.load(noise_path, sr=SAMPLE_RATE)
                if len(noise_full) < required_len:
                    continue
                    
                for i in range(SILENCE_SAMPLES_PER_NOISE_FILE):
                    start = random.randint(0, len(noise_full) - required_len)
                    clip = noise_full[start:start + required_len]
                    
                    # Apply some augmentations to silence too
                    if random.random() < PHONE_RECORDING_PROBABILITY:
                        clip = simulate_phone_recording(clip, SAMPLE_RATE)
                    
                    if random.random() < EQ_PROBABILITY:
                        clip = apply_eq_augmentation(clip, SAMPLE_RATE)
                    
                    filename = f"silence_{i+1}_{os.path.splitext(os.path.basename(noise_path))[0]}.wav"
                    filepath = os.path.join(silence_dir, filename)
                    save_audio(filepath, clip, SAMPLE_RATE)
                    
            except Exception as e:
                print(f"Error processing noise file {noise_path}: {e}")

    # Create dataset metadata
    metadata = {
        'dataset_info': {
            'name': 'Enhanced Piano Note Recognition Dataset',
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION_SECONDS,
            'num_keys': 88,
            'key_range': 'A0 to C8',
            'total_notes': len(NOTES),
            'chord_types': len(CHORDS),
            'samples_per_item': SAMPLES_PER_FONT_ITEM,
            'augmentations': [
                'Phone recording simulation',
                'Pitch shifting',
                'Time stretching',
                'Dynamic range compression',
                'EQ augmentation',
                'Background noise mixing',
                'Reverb simulation'
            ]
        },
        'notes': NOTES,
        'chords': list(CHORDS.keys()),
        'velocities': VELOCITIES,
        'snr_range': SNR_DB_RANGE
    }
    
    with open(os.path.join(OUTPUT_DIR, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nEnhanced dataset generation complete!")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print(f"Structure matches your training script's expectations.")

if __name__ == '__main__':
    generate_enhanced_dataset()