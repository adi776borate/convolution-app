# Streamlit Signal Convolution Visualizer 🎛️✨

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-streamlit-app-url.streamlit.app) <br>
*(Replace the link above with your deployed Streamlit Cloud URL!)*

An interactive web application built with Streamlit to visualize the convolution of two signals commonly encountered in electrical engineering and signal processing.

![App Screenshot](link/to/your/screenshot.png) <br>
*(Replace this with a link to a screenshot of your app! Upload the screenshot to your repo or use an image hosting service.)*

## Overview

This application allows users to:

1.  Select or define two continuous-time signals, $f(t)$ and $g(t)$.
2.  Visualize the individual signals $f(t)$ and $g(t)$.
3.  Visualize their convolution $y(t) = (f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) d\tau$.

The focus is on providing an intuitive understanding of signals and the convolution operation within the electrical engineering domain, featuring $\LaTeX$ style mathematical rendering in plots.

## Features

*   **Signal Selection:**
    *   Choose from a library of standard signals (Unit Step, Pulses, Exponential, Ramp, Triangle, etc.).
    *   Define custom signals using mathematical expressions involving `t` and the unit step `u(t, shift)`.
    *   Define custom piecewise signals using standard mathematical notation or `u(t)` gating.
*   **Interactive Plotting:**
    *   Uses Plotly for interactive plots allowing zooming, panning, and hovering to inspect values.
    *   Displays $f(t)$, $g(t)$, and the resulting $y(t)$ in separate subplots.
    *   Clean dark mode background theme for plots.
*   **Convolution Calculation:**
    *   Performs numerical convolution using `numpy.convolve`.
    *   Handles the special case where one signal is the standard **Unit Step**, performing numerical integration (`numpy.cumsum`) instead, correctly representing $f(t) * u(t) = \int_{-\infty}^{t} f(\tau) d\tau$.
*   **Mathematical Rendering:** Displays signal definitions and plot labels using $\LaTeX$ notation via Plotly's MathJax support.
*   **Configurable Parameters:** Allows adjusting the time range (`t_min`, `t_max`) and time step (`dt`) for the simulation.

## Technologies Used

*   **Python 3**
*   **Streamlit:** For creating the interactive web application interface.
*   **NumPy:** For numerical computations, array manipulation, and core signal generation.
*   **SciPy:** For numerical integration and potentially interpolation (`interp1d`).
*   **Plotly:** For generating interactive and publication-quality plots.

## Setup and Local Execution

To run this application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Replace `your-username/your-repo-name` with your actual GitHub repo details)*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure your `requirements.txt` file is up-to-date)*

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

5.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1.  Use the **sidebar** on the left to configure simulation parameters (`t_min`, `t_max`, `dt`).
2.  Define **Signal 1 ($f(t)$)** and **Signal 2 ($g(t)$)** using one of the available methods:
    *   **Choose Standard Signal:** Select from the dropdown list.
    *   **Define using u(t):** Enter a Python/NumPy expression using `t` as the variable. Use `u(t, shift)` for the unit step $u(t-\text{shift})$ and `np.` for NumPy functions (e.g., `np.exp`, `np.sin`). Remember to use `*` for multiplication (e.g., `2 * t`, not `2t`).
    *   **Define Piecewise:** Add segments defining the function's value over specific conditions on `t`. Use Python/NumPy syntax for conditions and values (e.g., `(t >= 0) & (t < 1)`).
3.  The plots for $f(t)$, $g(t)$, and the convolution $y(t)$ will update automatically.
4.  Interact with the plots (zoom, pan, hover) to explore the signals.

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to add features (like discrete-time convolution, more signals, Fourier analysis, etc.), feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Create a file named `LICENSE` in your repository and paste the text of the MIT License or another license of your choice into it)*

## Author

*   **Aditya Borate**

---

Made with Streamlit, NumPy, and Plotly.