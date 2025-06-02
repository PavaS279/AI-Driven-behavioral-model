import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title and description
st.title("Streamlit Sample Dashboard")
st.write("This is a basic Streamlit app demonstrating interactivity and data visualization.")

# Sidebar input
st.sidebar.header("User Input")
num_points = st.sidebar.slider("Number of data points", min_value=10, max_value=1000, value=100)

# Generate random data
data = pd.DataFrame({
    'x': np.arange(num_points),
    'y': np.random.randn(num_points).cumsum()
})

# Show data table
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(data)

# Line chart
st.subheader("Line Chart")
st.line_chart(data)

# Matplotlib plot
st.subheader("Matplotlib Plot")
fig, ax = plt.subplots()
ax.plot(data['x'], data['y'], color='skyblue', linewidth=2)
ax.set_title("Cumulative Random Data")
st.pyplot(fig)

# Input box
user_text = st.text_input("Enter a message", "Hello, Streamlit!")
st.write("You entered:", user_text)
