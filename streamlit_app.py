import streamlit as st
#from snowflake.snowpark

cnx = st.connection("snowflake")
session = cnx.session()

df = session.table("DF_MODEL_INPUT")
st.write(df)

df_model_input = df.to_pandas()
df_model_input

