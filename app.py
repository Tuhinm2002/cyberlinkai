import streamlit as st
from Multiapp import Multipage
from apps import _Classification_
from apps import home,_Regression_


def apps():
    app = Multipage()
    app.add_page("Home", home.app)
    app.add_page("Regression",_Regression_.app)
    app.add_page("Classification", _Classification_.app)
    app.run()
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





