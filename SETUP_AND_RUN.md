# LCU Project Setup and Running Instructions

## ✅ Fixed Issues

The code had the following issues that have been **FIXED**:

1. **Missing Dependencies**: Required packages (qiskit, qiskit-aer, numpy, matplotlib, pylatexenc) were not installed
   - **Fixed**: All packages have been installed in the virtual environment

2. **Broken Import Path** (in `lcu_applications.py`): The hardcoded path `/home/claude` was incorrect
   - **Fixed**: Changed to relative path that works cross-platform

## 🚀 How to Run

### Option 1: Run Algorithm.py (Main Program)
```powershell
cd "d:\CDAC Projects\LCU"
.\\.venv\Scripts\python.exe files\Algorithm.py
```

### Option 2: Run from VS Code Terminal
Simply open `files\Algorithm.py` in VS Code and press `Ctrl+F5` (Run without debugging) or select the Python interpreter terminal.

### Option 3: Direct Command
```powershell
"D:/CDAC Projects/LCU/.venv/Scripts/python.exe" "d:\CDAC Projects\LCU\files\Algorithm.py"
```

## 📊 Output

When you run `Algorithm.py`, you will see:
- Console output showing LCU run statistics
- A histogram of measurement outcomes (saved as `success_probability.png`)
- A quantum circuit diagram (saved as `lcu_circuit.png`)

## 📁 Project Structure

```
LCU/
├── .venv/                           # Virtual environment (auto-created)
├── 1D_infite_well_simulation.py    # (Currently empty)
├── files/
│   ├── Algorithm.py                 # Main LCU implementation ✓ WORKING
│   ├── lcu_improved.py              # Quantum gate definitions ✓ WORKING
│   ├── lcu_applications.py          # Advanced LCU applications ✓ WORKING
│   ├── lcu_documentation.md         # Documentation
│   └── README.md                    # Original README
└── SETUP_AND_RUN.md                # This file
```

## ✅ Verification

All code is now working and tested:
- ✅ `Algorithm.py` - Runs successfully and generates visualizations
- ✅ `lcu_improved.py` - Loads without errors
- ✅ `lcu_applications.py` - Import path fixed and functional

## 🔧 Dependencies Installed

- `qiskit` - Quantum computing framework
- `qiskit-aer` - Quantum simulator
- `numpy` - Numerical computing
- `matplotlib` - Plotting library
- `pylatexenc` - LaTeX encoding support

All are installed in: `d:\CDAC Projects\LCU\.venv`
