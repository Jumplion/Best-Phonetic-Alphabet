# Best Phonetic Alphabet

We've all has that problem when spelling out something where the other person hears an "M" when you said "N". That's why we have the [NATO Phonetic Alphabet](https://en.wikipedia.org/wiki/NATO_phonetic_alphabet) (Alfa, Bravo, Charlie, etc...). The thing is, the words in said phonetic alphabet must be distinct enough that they can't get confused with eachother. After all, having "Bet", "Debt", "Let", "Net", "Met", and "Set" all in the same phonetic alphabet wouldn't be very useful.

So this raises several question; what words would be best for a phonetic alphabet? Can we evaluate how good a set of words would be for a phonetic alphabet? 

## Resources, References, and Citations

* [Carnegie Mellon University (CMU) Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
* [International Phonetic Association](https://www.internationalphoneticassociation.org/)
	* [IPA Charts](https://www.internationalphoneticassociation.org/content/full-ipa-chart)
* [Interactive IPA Chart](https://www.ipachart.com/)
* [CoNNear cochlea model](https://github.com/HearingTechnology/CoNNear_cochlea)
* Wiki Pages
	* [Levenshtein Distance Algorithm (Edit Distance)](https://en.wikipedia.org/wiki/Levenshtein_distance)
	* [NATO Phonetic Alphabet]
	* [Phonetics]
	* [Phoneme]


## Overview

### What It Does
`BestPhoneticAlphabet.py` is the CLI entry point for the main interaction. Run `python src/BestPhoneticAlphabet.py` and you will be presented with a menu:

* `Generate Word Pair Scores`
	* Computes pairwise metrics for every unique word pair.
	* Writes letter-pair CSV files to `data/CSV_Files/word_pairs_<L1>_<L2>_data.csv`.
	* TODO: Write the CSV results into the DB so I don't have a bunch of huge .csv files lying around and can actually commit this for everyone to use properly.
* `Generate Word Averages`
	* Computes per-word aggregated statistics. 
	* Writes to `data/CSV_Files/word_averages.csv`.
	* NOTE: Requires that `Generate Word Pair Scores` be run first to aggregate the data.
* `Find Best (Randomized Trial)`
	* Runs `1000` trials of randomized  phonetic alphabets and scores them. 
	* Top candidates are shown and can be saved to `phoneme_data.db`.
	* Number of trials/candidates generated can be configured via `user_settings.json`
* `Score Premade Alphabet`
	* Run the scoring function on a predefined list (NATO by default) to get a comparable score.

### Scoring Methodology

Considering comparing A:B is the same as comparing B:A, the number of combinations we have to compare given a full list of 26 words would be;

$$C(26,2)=\binom{26}{2} = \frac{26!}{2!(26-2)!} = 325 $$

With each of these comparisons, we take both words' list of phonemes courtesy of the CMU Pronunciation Dictionary and compare their first phonemes with eachother, their second phonemes with eachother, their third, and so on.

The scoring for a candidate alphabet is a weighted aggregation of several metrics. The default weights are defined in `src/scoring.py` and can be altered in the `user_settings.json`.

#### Scoring Metrics

- Levenshtein distance (orthographic
	- Counts edit operations between words
	- Larger total distance favors less confusion.
- Phoneme Coordinate Distance
	- Phonemes are quantized into coordinates based on their properties and stored in `PHONEME_COORDINATES` (see `src/phoneme_utils.py`). 
	- Distance is calculated between two phonemes' coordinates. 
- Phoneme Audio Distance
	- `PHONEME_DISTANCES` stores precomputed acoustic distances between phonemes (used as an additional separation term). Distances were calculated using the audio files found under `voice/Phoneme Voice Files`
- Shared Sequence Penalties
	- Penalizes shared phoneme n-grams (number of shared subsequences) and shared orthographic (letter) suffixes to discourage similar-sounding or visually-similar words.
- Rhyme Penalties
	- Penalties for rhyming endings when stress patterns indicate rhymes. (Probably doesn't actually work...)

```
Score = 
	(weight_levenshtein * total_levenshtein)
	+ (weight_phoneme * total_phoneme_coordinate_distance)
	+ (weight_audio_diversity * total_phoneme_audio_distance)
	- (weight_shared_seq * shared_sequence_penalty)
	- (weight_shared_suffix * shared_suffix_penalty)
	- (weight_rhyme * rhyme_penalty)
```

Higher score means more overall separation between words. Weights can be adjusted in code oin `user_settings.json`

### Edge Cases and Notes
- The code uses CMU dict entries and normalizes phonemes by stripping stress digits for some metrics.
- If two words have unequal numbers of phonemes, the remaining phonemes from the longer word 
- There is a case to be made that comparing both words' Nth phonemes with eachother is inadequate, and instead all phonemes in one word should be compared with all other phonemes in the other word. I have no idea if that would be better or not.

## Files and folders (high level)

- `src/BestPhoneticAlphabet.py` — CLI entrypoint and interactive menu. Orchestrates user actions (generate pairwise scores, compute per-word averages, run randomized trials to find candidate alphabets, score premade alphabets). Can also be imported; the top-level `main()` function launches the menu.
- `src/create_databases.py` — scripts to build and populate the SQLite database(s) under `data/` (for example `data/phoneme_data.db`). This script reads CMU dictionary data and inserts normalized phoneme entries used by the scoring routines.
- `src/csv_readers.py` — helper utilities to read generated CSV matrices and averages from `data/CSV_Files/`. Used by analysis scripts and any code that needs to reload previously computed metrics.
- `src/csv_writers.py` — functions that compute and write pairwise letter CSV matrices and per-word averages to `data/CSV_Files/`. Called by the CLI when generating word-pair and word-average outputs.
- `src/phoneme_utils.py` — phoneme-related utilities and constants (for example `PHONEME_COORDINATES`), normalization helpers (strip stress markers, canonicalize symbols), and small helpers used across scoring and CSV generation.
- `src/scoring.py` — core scoring logic and weight definitions.

## Troubleshooting and common tasks

- If a script fails looking for `phoneme_data.db`, make sure `data/phoneme_data.db` exists. Create it by running `python src/create_databases.py`.
- If CSV readers/writers complain about missing directories, ensure `data/CSV_Files/` exists and is writable.
- 

# Setting Up the Repo

This guide provides detailed instructions for setting up the project on a **Windows** system using a virtual environment and installing dependencies from a `requirements.txt` file.

## Quick setup (Windows)

1. Clone the repo and switch to the project directory.
2. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Create the SQLite database used by the scoring routines (this script will populate CMU dictionary entries):

```powershell
python src\create_databases.py
```

5. Run the interactive utility:

```powershell
python src\BestPhoneticAlphabet.py
```

Notes:
- The code expects data files and CSV output under `data/` (the repository has been reorganized to place DB and CSVs in `data/`).
- Audio files are stored under `voice/` (previously `Phoneme Voice Files/`). If you want to run notebooks that rely on audio (e.g., `notebooks/connear_notebook.ipynb`), make sure `voice/` is available to the notebook.


## Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.8 or higher**: Download and install from [python.org](https://www.python.org/downloads/). During installation, check the box to add Python to your PATH.
- **Git** (optional, for cloning the repository): Download from [git-scm.com](https://git-scm.com/downloads).
- A terminal like Command Prompt, PowerShell, or Windows Terminal.

Verify Python is installed by running:

```powershell
python --version
```

You should see output like `Python 3.x.x`. If not, ensure Python is added to your PATH.

## Setup Instructions

Follow these steps to set up the project on your Windows machine.

### 1. Clone or Download the Project

If the project is hosted in a Git repository, clone it:

```powershell
git clone https://github.com/<username>/Best-Phonetic-Alphabet.git
cd Best-Phonetic-Alphabet
```

Alternatively, download and unzip the project files to a folder (e.g., `C:\Users\YourName\Best-Phonetic-Alphabet`), then navigate to it in your terminal:

```powershell
cd C:\Users\YourName\Best-Phonetic-Alphabet
```
