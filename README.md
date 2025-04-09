# SinglePulse Library

This Python library is designed for evaluating ICP-MS (Inductively Coupled Plasma Mass Spectrometry) data with a large number of peaks. It is specifically tailored for Perkin Elmer instruments working with Syngistix software. The library provides tools for data processing, peak detection, background correction, and calibration.

## Features

- **Support for Syngistix and Nanomodul Files**: Handles both regular Syngistix files and Nanomodul files.
- **Peak Analysis**: Detects peaks, calculates peak widths, and integrates peak areas.
- **Background Correction**: Supports global and local background correction.
- **Calibration**: Provides tools for calibrating data using known standards.
- **Visualization**: Includes plotting capabilities for data and analysis results.
- **Customizable Parameters**: Allows users to adjust parameters like cycle time, Savitzky-Golay filter settings, and peak detection thresholds.

## File Overview

### `singlepulse.py`

This is the core library containing all methods for handling Syngistix files and performing data analysis. Key components include:

- `SinglePulse` class: Handles data processing, peak detection, and analysis.
- `FilmCalibration` class: Provides tools for calibrating data using known standards.
- Utility functions: Includes helper functions like `calc_diameter`, `extract_string`, and `generate_colors`.

### `analysis.ipynb`

This Jupyter Notebook serves as an environment for data exploration using the `SinglePulse` library. It demonstrates how to:

- Import and process data files.
- Perform peak detection and analysis.
- Visualize results.
- Export analysis results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/eliasfoisner/singlepulse
   cd singlepulse
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional dependencies for plotting large datasets:
   ```bash
   pip install plotly-resampler
   ```

## Usage

### Example Workflow

1. **Import the Library**:
   ```python
   import singlepulse as sp
   ```

2. **Load Data**:
   ```python
   file_path = "path/to/your/data.csv"
   pulse = sp.SinglePulse(file_path)
   ```

3. **Process Data**:
   ```python
   pulse.timescale(isotope="C/13", cycle_time=50e-6)
   pulse.savgol(isotope="C/13")
   pulse.peak_finding(isotope="C/13", threshold=50)
   ```

4. **Visualize Results**:
   ```python
   pulse.plot(isotope="C/13", peaks=True, integration=True)
   sp.SinglePulse.fig.show()
   ```

5. **Calibrate Data**:
   ```python
   standards = [sp.SinglePulse("standard1.csv"), sp.SinglePulse("standard2.csv")]
   calibration = sp.FilmCalibration(standards, spotsizes=[(10, 10, 150)], film_concentration_percent=15)
   calibration.reg_function(isotope="C/13")
   ```

### Running the Jupyter Notebook

Open `analysis.ipynb` in Jupyter Notebook or JupyterLab to explore and analyze your data interactively.

## Dependencies

- Python 3.7+
- pandas
- numpy
- scipy
- seaborn
- matplotlib
- plotly
- plotly-resampler

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This library was developed to streamline the evaluation of Single Pulse Responses (SPR) recorded on Perkin Elmer instruments. Special thanks to the contributors (Lukas Brunnbauer, David Ken Gibbs) for their efforts in creating this tool.