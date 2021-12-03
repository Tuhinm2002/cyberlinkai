import streamlit as st

class Multipage:
    def __init__(self):
        self.pages=[]

    def add_page(self,title,func):
        self.pages.append({"title":title,"function":func})
    def run(self):
        page=st.sidebar.selectbox("App Navigation",self.pages,format_func=lambda page:page["title"])

        page["function"]()





