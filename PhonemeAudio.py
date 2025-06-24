import os

# Scientific and Data Libraries
import numpy as np

from tqdm import tqdm

# Distance and Edit Libraries
from fastdtw import fastdtw

# Audio Processing
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import librosa
import soundfile as sf

PHONEME_COORDINATES = {   
    # VOWELS    
    # Vowel |  Backness (Front 0 / Central 0.5 / Back 1)     Height (Low [Open] 0 / Mid 0.5 / High [Close] 1)      Roundness (Rounded 0 / Unrounded 1)
    "AA":  (0,  1,      0,       0),    # ɑ             father
    "AE":  (0,  0,      0,       0),    # æ             cat
    "AH":  (0,  0.5,    0.5,     0),    # ʌ or ə        cut
    "AO":  (0,  1,      0.5,     1),    # ɔ`            caught
    "AW":  (0,  0.75,   0.5,     1),    # aʊ            cow
    "AX":  (0,  0.5,    0.5,     0),    # ə (schwa)     about
    "AY":  (0,  0.5,    0.5,     0),    # aɪ            my
    "EY":  (0,  0,      0.65,    0),    # e             they
    "EH":  (0,  0,      0.5,     0),    # ɛ             bed
    "ER":  (0,  0.5,    0.5,     0),    # ɚ or ɝ        her
    "IY":  (0,  0,      1,       0),    # i             see
    "IH":  (0,  0,      0.85,    0),    # ɪ             sit
    "OW":  (0,  1,      0.65,    1),    # o             go
    "OY":  (0,  0.5,    0.5,     0.5),  # ɔɪ            toy
    "UW":  (0,  1,      1,       1),    # u             too
    "UH":  (0,  1,      0.85,    1),    # ʊ             put
    
    # CONSONANTS
    # Consonant | Place of Articulation | Manner of Articulation | Voiced/Unvoiced
    # 0 = Bilabial, 0.2 = Labiodental, 0.4 = Dental, 0.6 = Alveolar, 0.8 = Velar, 1 = Glottal
    # 0 = Stop, 0.125 = Affricate, 0.25 = Fricative, 0.5 = Nasal, 0.75 = Lateral Liquid, 0.875 = Rhotic Liquid, 1 = Glide
    # 0 = Voiceless, 1 = Voiced
    
    # Stops
    "P":  (1, 0,    0,      0),  # voiceless bilabial stop              pat
    "B":  (1, 0,    0,      1),  # voiced bilabial stop                 bat
    "D":  (1, 0.4,  0,      1),  # voiced alveolar stop                 dog
    "T":  (1, 0.4,  0,      0),  # voiceless alveolar stop              top
    "K":  (1, 0.8,  0,      0),  # voiceless velar stop                 cat
    "G":  (1, 0.8,  0,      1),  # voiced velar stop                    go
    
    # Affricates
    "CH": (1, 0.6,  0.125,  0),  # voiceless postalveolar affricate     chip
    "JH": (1, 0.6,  0.125,  1),  # voiced postalveolar affricate        judge

    # Fricatives
    "F":  (1, 0.2,  0.25,   0),  # voiceless labiodental fricative      fish
    "V":  (1, 0.2,  0.25,   1),  # voiced labiodental fricative         van
    "TH": (1, 0.4,  0.25,   0),  # voiceless dental fricative           thin
    "DH": (1, 0.4,  0.25,   1),  # voiced dental fricative              then
    "S":  (1, 0.4,  0.25,   0),  # voiceless alveolar fricative         see
    "Z":  (1, 0.4,  0.25,   1),  # voiced alveolar fricative            zoo
    "SH": (1, 0.6,  0.25,   0),  # voiceless postalveolar fricative     she
    "ZH": (1, 0.6,  0.25,   1),  # voiced postalveolar fricative        measure
    "HH": (1, 1,    0.25,   0),  # voiceless glottal fricative          he

    # Nasals
    "M":  (1, 0,    0.5,    1),  # bilabial nasal                       me
    "N":  (1, 0.4,  0.5,    1),  # alveolar nasal                       no
    "NG": (1, 0.8,  0.5,    1),  # velar nasal                          sing

    # Liquids
    "L":  (1, 0.4,  0.75,   1),  # alveolar lateral liquid              leaf    
    "R":  (1, 0.4,  0.875,  1),  # alveolar rhotic liquid               red

    # Glides (approximants)
    "Y":  (1, 0.6,  1,      1),  # palatal glide    (IPA: /j/)          yes
    "W":  (1, 0,    1,      1)   # bilabial glide   (IPA: /w/)          we
}   


# ----------------------------
# AUDIO ANALYSIS FUNCTIONS
# ----------------------------

"""_summary_
Estimates when an audio clip is "silent"
"""
def estimate_silence_threshold(audio, sample_length_ms=100):
    samples = [audio[i:i+sample_length_ms].dBFS for i in range(0, len(audio), sample_length_ms)]
    quiet_samples = [s for s in samples if s != float('-inf')]
    return min(quiet_samples) - 5 if quiet_samples else -40

"""_summary_
Normalize the audio chunk to the target dBs.
"""
def normalize_to_target(chunk, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - chunk.dBFS
    return chunk.apply_gain(change_in_dBFS)

"""_summary_
Attempts to split up the main phoneme recording sessions into individual phoneme audio clips
"""
def segment_phoneme_sessions(folder_path="Phoneme Voice Files", min_silence_len=10):
    
    for filename in tqdm(os.listdir(folder_path), desc="Processing Phoneme Audio Files", unit="file", colour="blue"):
        if not filename.lower().endswith(".wav"):
            continue

        phoneme = os.path.splitext(filename)[0].upper()
        file_path = os.path.join(folder_path, filename)

        try:
            audio = AudioSegment.from_wav(file_path)
            silence_thresh = estimate_silence_threshold(audio)

            chunks = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=-40,
                keep_silence=5,
                seek_step=1
            )

            if not chunks:
                print(f"⚠️ No utterances detected for {phoneme}")
                continue

            out_dir = os.path.join(folder_path, phoneme)
            os.makedirs(out_dir, exist_ok=True)

            for i, chunk in tqdm(enumerate(chunks, 1), desc=f"Segmenting {phoneme} | Auto Silence Threshold {silence_thresh:.2f} dBFS", unit="segment", leave=False, colour="green"):
                normalized_chunk = normalize_to_target(chunk)
                out_path = os.path.join(out_dir, f"{phoneme}_{i:02d}.wav")
                normalized_chunk.export(out_path, format="wav")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

"""_summary_
For each phoneme folder inside base_folder:
- Loads all utterances
- Finds the utterance with the lowest total DTW distance to the others
- Saves it as {phoneme}_medoid.wav
"""
def find_medoid_phoneme_recordings(base_folder="Phoneme Voice Files", target_sr=22050, output_filename="medoid.wav"):
    for phoneme in tqdm(sorted(os.listdir(base_folder)), desc="Finding Medoid Phoneme Recordings", unit="phoneme", colour="blue"):
        folder = os.path.join(base_folder, phoneme)
        if not os.path.isdir(folder):
            continue

        wav_files = sorted([f for f in os.listdir(folder) if 
                            f.endswith(".wav") and not f.endswith(output_filename) and not f.endswith("average.wav") and not f.endswith("median.wav")])

        if not wav_files:
            print(f"⚠️ No WAV files found in {folder}")
            continue

        waveforms = []
        paths = []

        for f in tqdm(wav_files, desc=f"Loading {phoneme} Recordings", unit="file", leave=False, colour="green"):
            path = os.path.join(folder, f)
            y, _ = librosa.load(path, sr=target_sr)
            y = librosa.util.normalize(y)
            waveforms.append(y)
            paths.append(path)

        n = len(waveforms)
        distances = np.zeros((n, n))

        # Compute pairwise DTW distances
        for i in range(n):
            for j in range(i + 1, n):
                dist, _ = fastdtw(waveforms[i], waveforms[j], dist=lambda x, y: np.abs(x - y))
                distances[i, j] = distances[j, i] = dist

        # Sum distances for each waveform
        totals = distances.sum(axis=1)
        best_idx = np.argmin(totals)

        # Save the medoid waveform as output
        best_wave = waveforms[best_idx]
        sf.write(os.path.join(folder, phoneme + "_" + output_filename), best_wave, target_sr)

"""_summary_
Resamples each phoneme's medoid.wav to match a fixed duration (default: 0.2 seconds).
Saves result as normalized.wav in the same folder.
"""
def normalize_phoneme_durations(base_folder="Phoneme Voice Files", target_duration=0.15, target_sr=22050, input_filename="_medoid.wav", output_filename="_normalized.wav"):
    for phoneme in PHONEME_COORDINATES.keys():
        folder = os.path.join(base_folder, phoneme)
        in_path = os.path.join(folder, f"{phoneme}{input_filename}")
        out_path = os.path.join(folder, f"{phoneme}{output_filename}")

        if not os.path.isfile(in_path):
            print(f"- Missing {phoneme}{input_filename} for {phoneme}")
            continue

        y, sr = librosa.load(in_path, sr=target_sr)
        duration = librosa.get_duration(y=y, sr=sr)
        stretch_factor = duration / target_duration

        # Skip if already within ~5% of target duration
        if 0.95 <= stretch_factor <= 1.05:
            sf.write(out_path, y, sr)
            continue

        # Time-stretch to match target duration
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
        sf.write(out_path, y_stretched, sr)

"""_summary_
Synthesizes a word audio file by concatenating medoid phoneme .wav files.
Parameters:
- word: the word string (e.g. "apple")
- phonemes: list of ARPAbet phonemes (e.g. ["AE", "P", "AH", "L"])
- phoneme_folder: where medoid.wav files are stored (subfolders by phoneme)
- output_base: base folder to write word .wav file into
"""
def synthesize_word_audio(word, phoneme_list, phoneme_folder="Phoneme Voice Files", output_base="Word Voicings", phoneme_file_suffix="_normalized.wav", crossfade_ms=20, envelope_ms=10):
    # Ensure output directory exists
    letter = word[0].upper()
    output_dir = os.path.join(output_base, letter)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{word}.wav")

    # Combine phoneme audio files
    combined = None
    for ph in phoneme_list:
        ph_folder = os.path.join(phoneme_folder, ph)
        ph_file = os.path.join(ph_folder, f"{ph}{phoneme_file_suffix}")
        if not os.path.exists(ph_file):
            print(f"Warning: Phoneme audio file not found: {ph_file}")
            continue
        ph_audio = AudioSegment.from_wav(ph_file)
        ph_audio = ph_audio.fade_in(envelope_ms).fade_out(envelope_ms)
        if combined is None:
            combined = ph_audio
        else:
            combined = combined.append(ph_audio, crossfade=crossfade_ms)

    # Export the combined audio
    # Normalize the final audio for consistent volume
    combined = effects.normalize(combined)
    combined.export(output_path, format="wav")
    return output_path
