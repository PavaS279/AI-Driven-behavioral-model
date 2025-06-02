import streamlit as st
#from snowflake.snowpark

cnx = st.connection("snowflake")
session = cnx.session()

df = session.table("DF_MODEL_INPUT")
st.write(df)
st.title("Hello, World!")
st.write("Welcome to your first Streamlit app.")
