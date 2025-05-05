# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signals import (STANDARD_SIGNALS, parse_signal_with_ut,
                     parse_piecewise_mathematical, u)


# --- Core Functions ---
def compute_convolution(sig1, sig2, dt):
    """Computes the convolution of two signals using numpy."""
    convolution = np.convolve(sig1, sig2) * dt
    return convolution

def integrate_signal(sig, t, dt):
    """
    Calculates the integral approximation from t_min up to each point in t.
    This approximates integral(-inf to t) for signals defined over [t_min, t_max].
    Uses cumulative sum.
    """
    integral = np.cumsum(sig) * dt
    return integral


def get_time_vector(t_min, t_max, dt):
    """Generates a time vector using linspace."""
    if t_max <= t_min or dt <= 0:
        return np.array([]) # Return empty if invalid params
    num_points = int(np.round((t_max - t_min) / dt)) + 1
    return np.linspace(t_min, t_max, num_points)

# --- Streamlit App UI ---
st.set_page_config(layout="wide")

st.title(r"Signal Convolution Visualizer $\LaTeX$")
st.markdown(r"""
Visualize time-domain signal convolutions:

  $$
  y(t) = (f * g)(t) = \int_{-\infty}^{\infty} f(\tau)\,g(t - \tau)\,d\tau
  $$
- Define custom signals using:
  - Standard signal templates (e.g., unit step, ramp, pulse)
  - Gated expressions using $u(t)$ (unit step function)
  - Piecewise definitions
""")

# --- Simulation Parameters ---
st.sidebar.header("Simulation Parameters")
t_min_default, t_max_default, dt_default = -5.0, 10.0, 0.01
t_min = st.sidebar.number_input(r"$t_{min}$", value=t_min_default, step=0.5)
t_max = st.sidebar.number_input(r"$t_{max}$", value=t_max_default, step=0.5)
dt = st.sidebar.number_input(r"$\Delta t$ (step size)", value=dt_default, format="%.4f", step=0.001, min_value=1e-6)
if t_max <= t_min:
    st.sidebar.error(r"$t_{max}$ must be greater than $t_{min}$")
    st.stop()
if dt <= 0:
    st.sidebar.error(r"$\Delta t$ must be positive")
    st.stop()
t = get_time_vector(t_min, t_max, dt)
if t.size == 0:
    st.error("Invalid time parameters.")
    st.stop()


# --- Signal Selection & Definition ---
st.sidebar.header("Signal Definitions")
signal_options = ["Choose Standard Signal", "Define using u(t)", "Define Piecewise (Math Notation)"]
standard_signal_names = list(STANDARD_SIGNALS.keys())
sig1 = np.zeros_like(t); sig2 = np.zeros_like(t)
sig1_latex_str = "0"; sig2_latex_str = "0"
sig1_name = ""; sig2_name = ""
# --- Signal 1 ---
st.sidebar.subheader(r"Signal 1: f(t)")
sig1_choice = st.sidebar.selectbox("Select Method", signal_options, key="sig1_choice")
if sig1_choice == "Choose Standard Signal":
    sig1_name = st.sidebar.selectbox("Standard Signal", standard_signal_names, key="sig1_std")
    sig1_func = STANDARD_SIGNALS[sig1_name]
    try: sig1 = sig1_func(t); sig1_latex_str = rf"\text{{{sig1_name}}}"
    except Exception as e: st.sidebar.error(f"Error generating {sig1_name}: {e}"); sig1 = np.zeros_like(t)
elif sig1_choice == "Define using u(t)":
    sig1_expr = st.sidebar.text_area("Enter function of t (use u(t, shift) for step)", "np.exp(-2*t) * u(t, 0)", key="sig1_ut_expr")
    if sig1_expr.strip():
        sig1 = parse_signal_with_ut(sig1_expr, t)
        sig1_latex_str = sig1_expr.replace('*',' \cdot ').replace('u(t, ','u(t-').replace('np.','')
    else: sig1_latex_str = "0"; sig1 = np.zeros_like(t)
else: # Define Piecewise
    st.sidebar.markdown("**Define $f(t)$ piecewise:** (Condition, Value)")
    if 'sig1_pieces' not in st.session_state:
        st.session_state.sig1_pieces = [("(t >= -1) & (t < 0)", "t + 1"), ("(t >= 0) & (t < 1)", "1 - t"), ("(t < -1) & (t >= 1)", "0")]
    pieces_data = []
    cols = st.sidebar.columns([3, 3, 1])
    for i, (cond, val) in enumerate(st.session_state.sig1_pieces):
        with cols[0]: new_cond = st.text_input(f"Cond {i+1}", cond, key=f"sig1_cond_{i}", label_visibility="collapsed")
        with cols[1]: new_val = st.text_input(f"Val {i+1}", val, key=f"sig1_val_{i}", label_visibility="collapsed")
        with cols[2]:
            if st.button("✖", key=f"sig1_rem_{i}", help="Remove piece"): st.session_state.sig1_pieces.pop(i); st.rerun()
            else: pieces_data.append((new_cond, new_val))
    st.session_state.sig1_pieces = pieces_data
    col_add, col_clear = st.sidebar.columns(2)
    if col_add.button("Add Piece", key="sig1_add"): st.session_state.sig1_pieces.append(("", "")); st.rerun()
    if col_clear.button("Clear All", key="sig1_clear"): st.session_state.sig1_pieces = []; st.rerun()
    if st.session_state.sig1_pieces:
         sig1 = parse_piecewise_mathematical(st.session_state.sig1_pieces, t)
         latex_parts = []
         for cond, val in [p for p in st.session_state.sig1_pieces if p[0].strip() and p[1].strip()]:
             val_disp = val.replace('*',' \cdot ').replace('np.','')
             cond_disp = cond.replace('&',' \land ').replace('|', ' \lor ').replace('<',' < ').replace('>',' > ').replace('<=',' \le ').replace('>=',' \ge ').replace('==',' = ').replace('!=',r' \ne ')
             latex_parts.append(rf"{val_disp} & \text{{if }} {cond_disp}")
         if latex_parts: sig1_latex_str = r"\begin{cases} " + r" \\ ".join(latex_parts) + r" \end{cases}"
         else: sig1_latex_str = "(\text{incomplete definition})"; sig1 = np.zeros_like(t)
    else: sig1_latex_str = "0"; sig1 = np.zeros_like(t)

# --- Signal 2 ---
st.sidebar.subheader(r"Signal 2: g(t)")
sig2_choice = st.sidebar.selectbox("Select Method", signal_options, key="sig2_choice", index=0)
if sig2_choice == "Choose Standard Signal":
    sig2_name = st.sidebar.selectbox("Standard Signal", standard_signal_names, key="sig2_std", index=0)
    sig2_func = STANDARD_SIGNALS[sig2_name]
    try: sig2 = sig2_func(t); sig2_latex_str = rf"\text{{{sig2_name}}}"
    except Exception as e: st.sidebar.error(f"Error generating {sig2_name}: {e}"); sig2 = np.zeros_like(t)
elif sig2_choice == "Define using u(t)":
    sig2_expr = st.sidebar.text_area("Enter function of t (use u(t, shift) for step)", "u(t, 0) - u(t, 1)", key="sig2_ut_expr")
    if sig2_expr.strip():
        sig2 = parse_signal_with_ut(sig2_expr, t)
        sig2_latex_str = sig2_expr.replace('*',' \cdot ').replace('u(t, ','u(t-').replace('np.','')
    else: sig2_latex_str = "0"; sig2 = np.zeros_like(t)
else: # Define Piecewise
    st.sidebar.markdown("**Define $g(t)$ piecewise:** (Condition, Value)")
    if 'sig2_pieces' not in st.session_state:
        st.session_state.sig2_pieces = [("t < 1", "0"), ("(t>=1) & (t<3)", "1"), ("t>=3","0")]
    pieces_data_g = []
    cols_g = st.sidebar.columns([3, 3, 1])
    for i, (cond, val) in enumerate(st.session_state.sig2_pieces):
        with cols_g[0]: new_cond = st.text_input(f"Cond {i+1}", cond, key=f"sig2_cond_{i}", label_visibility="collapsed")
        with cols_g[1]: new_val = st.text_input(f"Val {i+1}", val, key=f"sig2_val_{i}", label_visibility="collapsed")
        with cols_g[2]:
            if st.button("✖", key=f"sig2_rem_{i}", help="Remove piece"): st.session_state.sig2_pieces.pop(i); st.rerun()
            else: pieces_data_g.append((new_cond, new_val))
    st.session_state.sig2_pieces = pieces_data_g
    col_add_g, col_clear_g = st.sidebar.columns(2)
    if col_add_g.button("Add Piece", key="sig2_add"): st.session_state.sig2_pieces.append(("", "")); st.rerun()
    if col_clear_g.button("Clear All", key="sig2_clear"): st.session_state.sig2_pieces = []; st.rerun()
    if st.session_state.sig2_pieces:
        sig2 = parse_piecewise_mathematical(st.session_state.sig2_pieces, t)
        latex_parts = []
        for cond, val in [p for p in st.session_state.sig2_pieces if p[0].strip() and p[1].strip()]:
             val_disp = val.replace('*',' \cdot ').replace('np.','')
             cond_disp = cond.replace('&',' \land ').replace('|', ' \lor ').replace('<',' < ').replace('>',' > ').replace('<=',' \le ').replace('>=',' \ge ').replace('==',' = ').replace('!=',r' \ne ')
             latex_parts.append(rf"{val_disp} & \text{{if }} {cond_disp}")
        if latex_parts: sig2_latex_str = r"\begin{cases} " + r" \\ ".join(latex_parts) + r" \end{cases}"
        else: sig2_latex_str = "(\text{incomplete definition})"; sig2 = np.zeros_like(t)
    else: sig2_latex_str = "0"; sig2 = np.zeros_like(t)

# --- Perform Convolution ---
sig1_is_standard_step = (sig1_choice == "Choose Standard Signal" and sig1_name == "Unit Step")
sig2_is_standard_step = (sig2_choice == "Choose Standard Signal" and sig2_name == "Unit Step")
convolution_result = np.array([])
t_conv = t
t_conv_start = t_min
t_conv_end = t_max
convolution_title = r'Convolution: y(t) = (f * g)(t)'
plot_result = True
if sig1_is_standard_step and sig2_is_standard_step:
    convolution_result = t * u(t, 0); t_conv = t
elif sig1_is_standard_step:
    if np.any(sig2): convolution_result = integrate_signal(sig2, t, dt); t_conv = t
    else: convolution_result = np.zeros_like(t); t_conv = t
elif sig2_is_standard_step:
    if np.any(sig1): convolution_result = integrate_signal(sig1, t, dt); t_conv = t
    else: convolution_result = np.zeros_like(t); t_conv = t
elif np.any(sig1) and np.any(sig2):
    convolution_result = compute_convolution(sig1, sig2, dt)
    conv_len = len(convolution_result)
    t_conv_start = t_min + t_min
    t_conv_end = t_conv_start + (conv_len - 1) * dt
    t_conv = np.linspace(t_conv_start, t_conv_end, conv_len)
else:
    conv_len = len(t) + len(t) - 1
    convolution_result = np.zeros(conv_len)
    t_conv_start = t_min + t_min
    t_conv_end = t_max + t_max
    t_conv = np.linspace(t_conv_start, t_conv_end, conv_len)
    plot_result = np.any(sig1) or np.any(sig2)


# --- Plotting with Plotly ---  
st.subheader("Plots")

# Display the functions using st.latex
st.latex(f"f(t) = {sig1_latex_str}")
st.latex(f"g(t) = {sig2_latex_str}")

# Create Plotly figure with 3 rows of subplots
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=False, # Keep x-axes independent
    vertical_spacing=0.15, # Adjust spacing between plots
    subplot_titles=(r"Signal 1: f(t)", r"Signal 2: g(t)", convolution_title) 
)

# Add Signal 1 trace
fig.add_trace(go.Scatter(x=t, y=sig1, mode='lines', name='f(t)', line=dict(color='royalblue')),
              row=1, col=1)

# Add Signal 2 trace
fig.add_trace(go.Scatter(x=t, y=sig2, mode='lines', name='g(t)', line=dict(color='darkorange')),
              row=2, col=1)

# Add Convolution Result trace
# Only add if there's actually data to plot
if plot_result and convolution_result.size > 0:
    fig.add_trace(go.Scatter(x=t_conv, y=convolution_result, mode='lines', name='y(t)', line=dict(color='green')),
                  row=3, col=1)
else:
     # If result is zero, add a dummy invisible trace to keep the subplot structure
     # Or you could add an annotation, but plotting the zero line is often fine.
     fig.add_trace(go.Scatter(x=t_conv, y=convolution_result, mode='lines', name='y(t)=0', line=dict(color='lightgrey')), # Plot the zero line
                   row=3, col=1)


# Update layout for all subplots
fig.update_layout(
    height=750, # Adjust overall height
    showlegend=False, # Legend is redundant with subplot titles
    margin=dict(l=40, r=20, t=80, b=40), # Adjust margins
    hovermode='x unified' # Show unified hover info across y-values for a given x

)

# Update axes labels and ranges for each subplot specifically
fig.update_xaxes(title_text=r"t", range=[t_min, t_max], showgrid=True, row=1, col=1)
fig.update_yaxes(title_text=r"f(t)", showgrid=True, row=1, col=1)

fig.update_xaxes(title_text=r"t", range=[t_min, t_max], showgrid=True, row=2, col=1)
fig.update_yaxes(title_text=r"g(t)", showgrid=True, row=2, col=1)

fig.update_xaxes(title_text=r"t", range=[t_conv_start, t_conv_end], showgrid=True, row=3, col=1)
fig.update_yaxes(title_text=r"y(t)", showgrid=True, row=3, col=1)


# Display the Plotly figure in Streamlit
st.plotly_chart(fig, use_container_width=True)

# --- End Plotting with Plotly ---

st.markdown("---")
st.markdown("Made with 🤍 by Aditya Borate")