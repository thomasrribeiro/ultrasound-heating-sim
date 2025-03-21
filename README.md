# Transcranial Ultrasound Heating Simulation
Predicts the temperature rise for an ultrasound transducer transcranially.

## Getting started

### Installation

1. Create a virtual environment using [uv](https://github.com/astral-sh/uv):

```bash
# Create a virtual environment in .venv directory
uv venv

# Activate the virtual environment
source .venv/bin/activate
```

2. Install dependencies:

```bash
uv pip install -r requirements.txt
```

### Running the simulation

You can run the simulation in different modes:

```bash
# Run full simulation (acoustic + heating)
python run.py
```