
import streamlit as st
import numpy as np
import pandas as pd
import time

# Interactive Widget
st.write("Example of Streamlit interactive widget")
st.button("Click button bellow")
if st.button("Say Hello"):
    st.write('You clicked button 1')
else:
    st.write('Goodbye')
streamlit "Hello"