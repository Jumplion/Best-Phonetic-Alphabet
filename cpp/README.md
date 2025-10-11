# Word-Pair Scorer (C++)

This folder contains a small CMake-based C++ project to score word pairs and write results into an SQLite database.

Prerequisites

- CMake >= 3.16
- A C++17-capable compiler (MSVC/Clang/GCC)
- SQLite3 development libraries (on Windows, the `sqlite3.lib` and headers must be available)

Notes for Windows/MSVC users

- If CMake detects an SQLite library installed for MinGW/MSYS but you are using an MSVC generator, the build will fail with confusing include errors. In that case either:
- Install a SQLite3 library built for MSVC and pass its paths to CMake via `-DSQLITE3_LIBRARY=... -DSQLITE3_INCLUDE_DIR=...`, or
- Use a MinGW/MSYS generator (e.g., "MinGW Makefiles") to match the detected SQLite installation.

Example of passing SQLite paths to CMake (PowerShell):

```powershell
cmake .. -DSQLITE3_LIBRARY="C:/libs/sqlite3/lib/sqlite3.lib" -DSQLITE3_INCLUDE_DIR="C:/libs/sqlite3/include" -G "Visual Studio 17 2022"
```

Build (PowerShell)

```powershell
cd "f:\Repos\Best Phonetic Alphabet\cpp"
mkdir build; cd build
cmake .. -G "Visual Studio 16 2019"  # or the generator matching your VS
cmake --build . --config Release
```

Run

```powershell
# from the build output dir (e.g. Debug/Release)
.\wordpair_scorer.exe ..\..\phoneme_data.db
```

Notes

- The scoring implementation in `src/scorer.cpp` is a placeholder. Replace with your phoneme-based scoring logic.
- `DBWriter` uses the SQLite C API directly. The table `word_pair_scores` will be created automatically if it doesn't exist.

Helper: bootstrap vcpkg and install sqlite3

-----------------------------------------
There's a helper PowerShell script to bootstrap `vcpkg` and install `sqlite3`. From the `cpp` folder run (PowerShell):

```powershell
.\scripts\setup_vcpkg.ps1
```

Then configure CMake from a clean build directory using the vcpkg toolchain file:

```powershell
mkdir build; cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -A x64 -G "Visual Studio 17 2022"
cmake --build . --config Release
```
