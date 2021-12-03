import streamlit as st
from Multiapp import Multipage
from apps import _Classification_
from apps import home,_Regression_


def welcome():
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)



#from apps import car_price_predictor

def apps():
    app = Multipage()
    app.add_page("Home", home.app)
    app.add_page("Regression",_Regression_.app)
    app.add_page("Classification", _Classification_.app)
    app.run()
welcome()
apps()
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")




# st.image()





