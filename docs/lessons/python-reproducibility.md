# Python and reproducible AI work

**Prerequisites:** [How learning works](learning-loop.md)  
**Outcome:** You can create an isolated Python environment and run a small, reproducible experiment.

## Why setup is part of AI engineering

AI work depends on numerical libraries, notebooks, datasets, model files, and hardware-specific runtimes. If the environment is implicit, you cannot reproduce a result or understand whether a change came from your code, your data, or a dependency update.

The first rule is simple:

> Make the environment boring, isolated, and written down.

## Setup

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib jupyter
```

Use `python -m pip`, not a random global `pip`, so the command is attached to the interpreter you intend to use.

## Your first reproducible experiment

```python
from random import Random

rng = Random(7)
values = [rng.random() for _ in range(5)]
print([round(value, 3) for value in values])
```

The seed makes this small experiment repeatable. Reproducibility does not mean every real-world process is deterministic; it means you know which choices you controlled.

## Android bridge

This is similar to Gradle version catalogs, build variants, and lockfiles. A build that works only on one laptop is not a stable build. A model experiment that works only in one notebook session is not a trustworthy experiment.

## Exercises

1. Run the code twice. Do the values match?
2. Change the seed to `8`. What changes?
3. Write down Python, NumPy, pandas, and Matplotlib versions in your notes.

**Next topic:** numerical computing with Python, NumPy, and array shapes.

