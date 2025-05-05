# signals.py
import numpy as np
import streamlit as st
import re # For parsing user input

# --- Helper Function: Unit Step ---
# Vectorized unit step function u(t-shift)
def u(t, shift=0):
    """Vectorized Unit Step function u(t-shift)."""
    return np.where(t >= shift, 1.0, 0.0)

# --- Standard Signal Definitions ---
# These functions take a time vector 't' and return the signal array 'y'

def unit_step(t):
    """Standard Unit Step Signal U(t)"""
    return u(t)

def pulse_1sec_2high(t):
    """1 Sec Pulse, Amplitude 2"""
    return 2 * (u(t) - u(t - 1))

def pulse_2_5sec(t):
    """2.5 Sec Pulse, Amplitude 1"""
    return u(t) - u(t - 2.5)

def pulse_2sec(t):
    """2 Sec Pulse, Amplitude 1"""
    return u(t) - u(t - 2)

def narrow_pulse(t, width=0.1):
    """Narrow Pulse (approx. Dirac), Amplitude 1/width"""
    center = width / 2
    return (1.0/width) * (u(t - center + width/2) - u(t - center - width/2))

def exponential_decay(t, alpha=1):
    """Exponential Decay e^(-alpha*t) * u(t)"""
    return np.exp(-alpha * t) * u(t)

def fast_exponential_decay(t):
    """Fast Exponential Decay e^(-3*t) * u(t)"""
    return exponential_decay(t, alpha=3)

def very_fast_exponential_decay(t):
    """Very Fast Exponential Decay e^(-10*t) * u(t)"""
    return exponential_decay(t, alpha=10)

def biphasic_pulse(t, width=0.5):
    """Simple Biphasic Pulse"""
    return (u(t) - u(t - width)) - (u(t - width) - u(t - 2 * width))

def ramp(t):
    """Ramp Function t * u(t) ONLY active between t=0 and t=1."""
    # t * gate_on_at_0 * gate_off_at_1
    return t * (u(t) - u(t - 1))

def inverse_ramp(t): # Removed t_max parameter as it's no longer needed
    """Inverse Ramp Function (1-t) ONLY active between t=0 and t=1."""
    # (1-t) * gate_on_at_0 * gate_off_at_1
    return (1 - t) * (u(t) - u(t - 1)) 

def triangle(t, width=2):
    """Triangle Pulse centered at width/2"""
    # Ramp up from 0 to width/2, ramp down from width/2 to width
    return (t * (u(t) - u(t - width/2)) +
            (width - t) * (u(t - width/2) - u(t - width)))

def impulse(t, tolerance=1e-3):
    """Approximation of Dirac Impulse delta(t) using a very narrow pulse"""
    # Find the index closest to t=0
    zero_idx = np.argmin(np.abs(t))
    imp = np.zeros_like(t)
    # Assign a large value at t=0 (or closest point)
    # The magnitude should ideally integrate to 1. Depends on dt.
    if np.abs(t[zero_idx]) < tolerance:
         # Approximate based on dt: Area = height * dt => height = 1/dt
        dt = t[1] - t[0] if len(t) > 1 else 1
        imp[zero_idx] = 1.0 / dt if dt > 0 else 1.0
    return imp

def echo_signal(t, delay=2.0, decay=0.5):
    """An example signal with an echo: pulse + delayed decayed pulse"""
    original_pulse = pulse_1sec_2high(t)
    echo_pulse = decay * pulse_1sec_2high(t - delay)
    return original_pulse + echo_pulse

def pulse_train_signal(t, pulse_width=0.5, period=2.0, num_pulses=3):
    """A train of rectangular pulses"""
    signal = np.zeros_like(t)
    for n in range(num_pulses):
        start_time = n * period
        signal += u(t - start_time) - u(t - start_time - pulse_width)
    return signal

def damped_sine(t, freq=1.0, damping=0.5):
    """Damped Sine Wave: e^(-damping*t) * sin(2*pi*freq*t) * u(t)"""
    return np.exp(-damping * t) * np.sin(2 * np.pi * freq * t) * u(t)

def oddball_signal(t):
    """An arbitrary example combining functions"""
    return (np.sin(2 * np.pi * 0.5 * t) * (u(t) - u(t-3)) +
            2 * exponential_decay(t-3, alpha=2))


# --- Dictionary of Standard Signals ---
STANDARD_SIGNALS = {
    "Unit Step": unit_step,
    "1 Sec Pulse (Amp=2)": pulse_1sec_2high,
    "2.5 Sec Pulse (Amp=1)": pulse_2_5sec,
    "2 Sec Pulse (Amp=1)": pulse_2sec,
    "Narrow Pulse (Approx. Impulse)": narrow_pulse,
    "Exponential Decay (a=1)": exponential_decay,
    "Fast Exp Decay (a=3)": fast_exponential_decay,
    "Very Fast Exp Decay (a=10)": very_fast_exponential_decay,
    "BiPhasic Pulse": biphasic_pulse,
    "Ramp": ramp,
    "Inverse Ramp": inverse_ramp,
    "Triangle Pulse (width=2)": triangle,
    "Impulse (Approx. Delta)": impulse,
    "Echo Signal": echo_signal,
    "Pulse Train": pulse_train_signal,
    "Damped Sine": damped_sine,
    "Oddball Example": oddball_signal,
}

# --- User Defined Signal Parsing ---

def parse_signal_with_ut(expression_str, t):
    """
    Parses a user-defined signal expression containing u(t-a) notation.
    Uses eval() - BE CAREFUL WITH UNTRUSTED INPUT.
    """
    try:
        # Make numpy functions available, plus our unit step 'u'
        # Limit the available scope for eval for security
        safe_dict = {
            'np': np,
            'u': u,
            't': t,
            # Add other safe functions if needed: sin, cos, exp, etc.
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'pi': np.pi,
            'sqrt': np.sqrt,
            'log': np.log,
            'log10': np.log10,
            'abs': np.abs,
            'max': np.maximum, # Use np.maximum for element-wise max
            'min': np.minimum, # Use np.minimum for element-wise min
            'heaviside': np.heaviside # Alternative to u(t)
        }

        # Replace common function names if user types them differently
        expression_str = expression_str.replace('^', '**') # Allow ^ for power

        # Evaluate the expression
        y = eval(expression_str, {"__builtins__": {}}, safe_dict)

        # Ensure output is a numpy array of the same shape as t
        if not isinstance(y, np.ndarray) or y.shape != t.shape:
             # If eval returns a scalar, broadcast it to the shape of t
             if np.isscalar(y):
                 y = np.full_like(t, float(y))
             else:
                 raise ValueError("Expression did not evaluate to a valid array matching t.")
        return y.astype(float) # Ensure float type

    except Exception as e:
        st.error(f"Error evaluating function: {e}")
        st.warning("Make sure your expression uses 't' as the variable, 'u(t, shift)' for unit step (e.g., u(t, 5) for u(t-5)), and standard Python/NumPy math functions (sin, cos, exp, np, pi...). Example: 2 * u(t, 0) - 2 * u(t, 1)")
        return np.zeros_like(t) # Return zeros on error

def parse_piecewise_mathematical(pieces, t):
    """
    Generates a signal from mathematically defined piecewise segments.
    'pieces' is a list of tuples: [ (condition_str, value_str), ... ]
    Uses np.select which is safer than multiple evals.
    """
    conditions = []
    values = []
    error_occurred = False

    # Define the safe dictionary for evaluating conditions and values
    safe_dict = {
        'np': np,
        'u': u, # Allow u(t) in piecewise definitions too if needed
        't': t,
        'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'pi': np.pi,
        'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10, 'abs': np.abs,
        'max': np.maximum, 'min': np.minimum, 'heaviside': np.heaviside
    }

    for i, (cond_str, val_str) in enumerate(pieces):
        # Check if either string is empty or just whitespace before trying to eval
        if not cond_str.strip() or not val_str.strip():
            # Silently skip this incomplete piece during parsing
            continue

        try:
            # Evaluate the condition string (should result in a boolean array)
            cond_str_safe = cond_str.replace('^', '**')
            condition = eval(cond_str_safe, {"__builtins__": {}}, safe_dict)
            if not isinstance(condition, np.ndarray) or condition.dtype != bool or condition.shape != t.shape:
                # Try to evaluate as scalar condition
                if isinstance(condition, bool):
                    condition = np.full_like(t, condition, dtype=bool)
                else:
                    st.error(f"Error in piece {i+1}: Condition '{cond_str}' did not evaluate to a boolean array matching t.")
                    error_occurred = True
                    continue 

            # Evaluate the value string (should result in a numerical array or scalar)
            val_str_safe = val_str.replace('^', '**')
            value = eval(val_str_safe, {"__builtins__": {}}, safe_dict)
            if not isinstance(value, np.ndarray) or value.shape != t.shape:
                # If eval returns a scalar, broadcast it
                if np.isscalar(value):
                    value = np.full_like(t, float(value))
                else:
                    st.error(f"Error in piece {i+1}: Value '{val_str}' did not evaluate to a valid array/scalar.")
                    error_occurred = True
                    continue 

            conditions.append(condition)
            values.append(value.astype(float)) # Ensure float type

        except Exception as e:
            st.error(f"Error evaluating piece {i+1} ('{cond_str}' / '{val_str}'): {e}")
            st.warning(
                "Ensure conditions use 't' and result in True/False. "
                "For combined conditions like 'a < t < b', use NumPy's element-wise '&' (AND) operator: "
                "'(t > a) & (t < b)'. "
                "Ensure values use 't' or are constants (e.g., '0', 'np.sin(t)', 't**2')."
            )
            error_occurred = True


    if error_occurred and not conditions:
        return np.zeros_like(t) # Return zeros if all pieces failed

    if not conditions:
        # Check if there were non-empty pieces defined that failed, or if all were empty
        if any(p[0].strip() or p[1].strip() for p in pieces):
             st.warning("No valid piecewise segments could be parsed.")
        else:
             st.info("Define piecewise segments to generate the signal.") 
        return np.zeros_like(t)

    # np.select(conditions_list, values_list, default_value)
    return np.select(conditions, values, default=0.0)