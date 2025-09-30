import utils.session
import streamlit as st
import numpy as np
import os

# --- Page Config MUST be first ---
st.set_page_config(
    page_title="Neural Network From Scratch",
    page_icon="🧠",
    layout="centered"
)

# --- Activation Function ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Forward Propagation ---
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2

# --- Load Model Safely ---
@st.cache_data
def load_model():
    if not os.path.exists('model_params.npz'):
        st.warning("⚠️ model_params.npz not found! Using random weights.")
        W1 = np.random.randn(2,2)
        b1 = np.zeros((1,2))
        W2 = np.random.randn(2,1)
        b2 = np.zeros((1,1))
        return W1, b1, W2, b2
    params = np.load('model_params.npz')
    return params['W1'], params['b1'], params['W2'], params['b2']

W1, b1, W2, b2 = load_model()

# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info("""
🧠 Neural Network Demo  
- XOR dataset (2-2-1)  
- NumPy implementation  
- Forward & Backward Propagation  
- Streamlit UI
""")

# --- Main UI ---
st.title("🧠 Neural Network From Scratch")
st.markdown("This is a simple 2-2-1 neural network trained on the XOR dataset.")
st.divider()

st.subheader("Try it yourself:")

col1, col2 = st.columns(2)
with col1:
    x1 = st.selectbox("Input 1 (x1):", [0, 1])
with col2:
    x2 = st.selectbox("Input 2 (x2):", [0, 1])

if st.button("Predict Output"):
    X = np.array([[x1, x2]])
    pred = forward_propagation(X, W1, b1, W2, b2)
    result = int(pred > 0.5)
    st.success(f"🧩 Prediction: **{result}** (Output ~ {pred[0][0]:.3f})")

st.divider()
st.markdown("""
✅ Trained on XOR Dataset  
✅ Implemented Forward & Backward Propagation  
✅ Built only with NumPy  
✅ Deployed with Streamlit
""")
