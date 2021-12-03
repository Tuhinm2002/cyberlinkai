import streamlit as st

def app():
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.markdown("# Welcome To Cyberlink AI")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""# What we do 
    Click top left arrow to get started
    if you don't find app navigation
    """)
    # s=st.selectbox("Select",("Iris dataset","Fashion_mnist"))

    st.markdown("# Liked our work or you are interested in our projects")
    b1=st.button("About us")
    b2=st.button("Contact us")
    if b1:
        st.markdown("""# WE BASICALLY HELP BUILD ML MODELS WITHOUT CODING...
        IF ANYONE WANTS THE CODE OF THE MODEL THEY CAN DOWNLOAD IT""")
    if b2:
        st.markdown("""# email: tuhinm2002@gmail.com
        or tuhinm151@gmail.com""")





