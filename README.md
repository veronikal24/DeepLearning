# DeepLearning

## Poetry setup

This project uses Poetry for dependency management. To set up locally (PowerShell):

```powershell
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python -

# From the repo root
cd c:\git\DeepLearning

# Create virtual environment and install deps
poetry install

# Activate the virtual environment (Poetry-managed)
poetry shell

# Run the simple test (compiles the example script without executing heavy IO)
python -m py_compile veronika_test.py
```

If you prefer to pass paths when running `veronika_test.py`, call the script module or edit the constants inside the file.

### Running via Poetry

After `poetry install` you can either run the script directly with Python inside the Poetry environment:

```powershell
poetry run python veronika_test.py C:\path\to\in.csv C:\path\to\out.parquet
```

Or use the convenience script installed by Poetry:

```powershell
poetry run veronika-test [in.csv] [out.parquet]
```

Both approaches run the code inside Poetry's virtual environment where `pandas` and other deps are installed.
# DeepLearning