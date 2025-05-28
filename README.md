# Best-Phonetic-Alphabet

Welcome to Best-Phonetic-Alphabet! This guide provides detailed instructions for setting up the project on a **Windows** system using a virtual environment and installing dependencies from a `requirements.txt` file.

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

### 2. Create a Virtual Environment

A virtual environment isolates project dependencies from your system-wide Python installation.

Run the following command to create a virtual environment named `venv`:

```powershell
python -m venv venv
```

This creates a `venv` folder in your project directory (e.g., `Best-Phonetic-Alphabet\venv`).

### 3. Activate the Virtual Environment

Activate the virtual environment to use its isolated Python environment:

```powershell
venv\Scripts\Activate.ps1
```

After activation, your terminal prompt should change to show `(venv)`, like:

```
(venv) C:\Users\YourName\Best-Phonetic-Alphabet>
```

This indicates you’re now using the virtual environment’s Python and `pip`.

### 4. Install Dependencies

The project dependencies are listed in `requirements.txt`. Install them with:

```powershell
pip install -r requirements.txt
```

This installs all required packages into the virtual environment. If `requirements.txt` includes specific versions (e.g., `requests==2.28.1`), those exact versions will be installed.

To verify the packages are installed, run:

```powershell
pip list
```

You should see a list of installed packages matching `requirements.txt`.

### 5. Run the Project

Once dependencies are installed, you can run the project.

### 6. Deactivate the Virtual Environment

When you’re done working, deactivate the virtual environment:

```powershell
deactivate
```

This returns you to your system’s default Python environment, and the `(venv)` prefix will disappear from your prompt.

## Project Structure

Here’s an overview of the project’s directory:

```
Best-Phonetic-Alphabet/
├── venv/                   # Virtual environment (not tracked in Git)
├── requirements.txt        # List of project dependencies
├── BestPhoneticAlphabet.py # Main python file
├── README.md               # This file
└── ...                     # Other project files
```

## Updating Dependencies

If you install new packages during development (e.g., `pip install pandas`), update `requirements.txt` to reflect the changes:

```powershell
pip freeze > requirements.txt
```

This ensures others can replicate your environment.

## Troubleshooting

- **Command not found: python**: Ensure Python is installed and added to your PATH. Restart your terminal or computer if you just installed Python.
- **Permission errors**: If you encounter permission issues, try running your terminal as Administrator (right-click Command Prompt or PowerShell and select "Run as administrator").
- **Missing packages**: If `pip install -r requirements.txt` fails, ensure you’re in the virtual environment (`venv\Scripts\Activate.ps1`) and check for typos in `requirements.txt`.
- **Old Python version**: If `python --version` shows an outdated version, you may have multiple Python installations. Use `py -3 --version` or specify the path to the desired Python executable (e.g., `C:\Python39\python.exe`).

## Best Practices

- **Don’t commit `venv/`**: The `venv` folder is included in `.gitignore` to avoid committing it to version control. Share `requirements.txt` instead.
- **Keep `requirements.txt` updated**: Always run `pip freeze > requirements.txt` after installing or upgrading packages.
- **Test on a clean environment**: Periodically test your setup by creating a new virtual environment and installing dependencies from `requirements.txt` to ensure reproducibility.
- **Use specific versions**: The `requirements.txt` file pins package versions (e.g., `requests==2.28.1`) to avoid compatibility issues.

## Additional Resources

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Pip User Guide](https://pip.pypa.io/en/stable/user_guide/)
- [Managing Python on Windows](https://docs.python.org/3/using/windows.html)

If you encounter issues or have questions, feel free to reach out to the project maintainer!
