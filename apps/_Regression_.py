import streamlit as st
import pandas as pd
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,Ridge
from io import StringIO
import sys



def app():
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.markdown("# Regression ")

    selector=st.selectbox("Choose Machine Learning Library",("Sklearn","Tensorflow"),key=0)
    if selector=="Tensorflow":
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        string_make = """import pandas as pd
import tensorflow as tf 
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split"""
        selector1=st.selectbox("Choose Regressors",("Linear Regression","Logistic Regression"),key=1)
        if selector1=="Linear Regression":
            n_inputs=st.number_input("Enter the number of neurons you want to add",value=0,key=0)
            n_inputs1=st.number_input("Enter the number of dense layers you want to add",value=0,key=1)
            n_inputs2=st.number_input("Enter the learning rate of neurons you want to add",key=2)
            a=[]
            for i in range(n_inputs1):
                a.append(tf.keras.layers.Dense(n_inputs,activation="relu"))
            model=tf.keras.Sequential([j for j in a])
            model.add(tf.keras.layers.Dense(1))

            string_marker = """n_inputs=int(input("Enter the number of neurons you want to add"))\n
n_inputs1=int(input("Enter the number of dense layers you want to add"))\n
n_inputs2=int(input("Enter the learning rate of neurons you want to add"))\n
a=[]\n
for i in range(n_inputs1):\n
    a.append(tf.keras.layers.Dense(n_inputs,activation="relu"))\n
model=tf.keras.Sequential([j for j in a])\n
model.add(tf.keras.layers.Dense(1))\n"""


            selector2=st.selectbox("Select the type of cost function",("mae","mse","binary_cross_entropy"),key=2)
            selector3=st.selectbox("Select the type of optimizer",("sgd","ADAM","RMSprop"),key=3)


            apply=st.checkbox("Apply custom learning rate",key=0)
            if apply:
                if selector2=="mae":
                    if selector3=="sgd":
                        model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["mae"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["mae"])\n"""
                    elif selector3=="ADAM":
                        model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["mae"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["mae"])\n"""

                    elif selector3=="RMSprop":
                        model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["mae"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["mae"])\n"""
        
        
                elif selector2=="mse":
                    if selector3=="sgd":
                        model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["mse"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["mse"])\n"""

                    elif selector3=="ADAM":
                        model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["mse"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["mse"])\n"""

                    elif selector3=="RMSprop":
                        model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["mse"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["mse"])\n"""
        

                elif selector2=="binary_cross_entropy":
                    if selector3=="sgd":
                        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

                    elif selector3=="ADAM":
                        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

                    elif selector3=="RMSprop":
                        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

            else:
                if selector2 == "mae":
                    if selector3 == "sgd":
                        model.compile(loss=tf.keras.losses.mae,
                                      optimizer=tf.keras.optimizers.SGD(), metrics=["mae"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(), metrics=["mae"])\n"""

                    elif selector3 == "ADAM":
                        model.compile(loss=tf.keras.losses.mae,
                                      optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
                        string_marker = string_marker + """model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])\n"""

                    elif selector3 == "RMSprop":
                        model.compile(loss=tf.keras.losses.mae,
                                      optimizer=tf.keras.optimizers.RMSprop(), metrics=["mae"])
                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.RMSprop(), metrics=["mae"])\n"""


                elif selector2 == "mse":
                    if selector3 == "sgd":
                        model.compile(loss=tf.keras.losses.mse,
                                      optimizer=tf.keras.optimizers.SGD(), metrics=["mse"])
                        string_marker = string_marker + """model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.SGD(), metrics=["mse"])\n"""

                    elif selector3 == "ADAM":
                        model.compile(loss=tf.keras.losses.mse,
                                      optimizer=tf.keras.optimizers.Adam(), metrics=["mse"])
                        string_marker = string_marker + """model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.Adam(), metrics=["mse"])\n"""



                    elif selector3 == "RMSprop":
                        model.compile(loss=tf.keras.losses.mse,
                                      optimizer=tf.keras.optimizers.RMSprop(), metrics=["mse"])
                        string_marker = string_marker + """model.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.RMSprop(), metrics=["mse"])\n"""


                elif selector2 == "binary_cross_entropy":
                    if selector3 == "sgd":
                        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
                        string_marker = string_marker +"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])\n"""


                    elif selector3 == "ADAM":
                        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])\n"""


                    elif selector3 == "RMSprop":
                        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])

                        string_marker=string_marker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])\n"""

            rs_input=st.number_input("Enter the random state of the model",value=0,key=3)
            split_input=st.number_input("Enter the amount for testing samples",value=0,key=4)

            string_marker=string_marker+"""rs_input=int(input("Enter the random state of the model"))\n"""
            string_marker=string_marker+"""split_input=int(input("Enter the amount for testing samples"))\n"""

            files=st.file_uploader("upload files",type=["csv","xls","xlsx"])
            string_marker=string_marker+"""files=open()#enter the directory of your spreadsheet/dataset file\n"""
            if files is not None:
                if files.name.split(".")[1]=="csv":
                    data = pd.read_csv(files)
                    data = pd.DataFrame(data)
                    string_marker=string_marker+"""data = pd.read_csv(files)\n
data = pd.DataFrame(data)\n"""
                    st.dataframe(data)
                    col=st.text_input("Enter the target column for y value")
                    string_marker=string_marker+"""col=input("Enter the target column for y value")\n"""
                    if col:
                        moded=pd.get_dummies(data)
                        X=moded.drop(col,axis=1)
                        Y=moded[col]
                        x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=rs_input,test_size=(split_input/100))
                        string_marker=string_marker+"""moded=pd.get_dummies(data)\n
X=moded.drop(col,axis=1)\n
Y=moded[col]\n
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=rs_input,test_size=(split_input/100))\n"""

                        epochs_count=st.number_input("Enter the count of epochs",value=0,key=5)
                        string_marker=string_marker+"""epochs_count=int(input("Enter the count of epochs"))\n"""
                        if epochs_count:
                            model.fit(x_train,y_train,epochs=epochs_count)
                            y_predict=model.predict(x_test)
                            string_marker=string_marker+"""history=model.fit(x_train,y_train,epochs=epochs_count)\n
y_predict=model.predict(x_test)\n"""
                            st.write(y_predict)

                            string_total=string_make+"""\n"""+string_marker
                            text_edition=st.checkbox("Download .txt edition of the code ")
                            code_edition = st.checkbox("Download .py edition of the code ")
                            if text_edition:
                                st.download_button(label="Download",data=string_total,file_name="text/.txt")

                            elif code_edition:
                                st.download_button(label="Download",data=string_total,file_name="python/.py")
                                




                elif files.name.split(".")[1]=="xls":
                    data = pd.read_excel(files)
                    data = pd.DataFrame(data)
                    string_marker=string_marker+"""data = pd.read_excel(files)\n
data = pd.DataFrame(data)\n"""
                    st.dataframe(data)
                    col = st.text_input("Enter the target column for y value")
                    string_marker=string_marker+"""col = input("Enter the target column for y value")\n"""
                    if col:
                        moded = pd.get_dummies(data)
                        X = moded.drop(col, axis=1)
                        Y = moded[col]
                        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input,
                                                                            test_size=(split_input / 100))
                        string_marker=string_marker+"""moded = pd.get_dummies(data)\n
X = moded.drop(col, axis=1)
Y = moded[col]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input,test_size=(split_input / 100))\n"""

                        epochs_count = st.number_input("Enter the count of epochs", value=0, key=5)
                        string_marker=string_marker+"""epochs_count = int(input("Enter the count of epochs"))\n"""
                        if epochs_count:
                            model.fit(x_train, y_train, epochs=epochs_count)
                            y_predict = model.predict(x_test)
                            st.write(y_predict)

                            string_total = string_make + """\n""" + string_marker
                            text_edition = st.checkbox("Download .txt edition of the code ")
                            code_edition = st.checkbox("Download .py edition of the code ")
                            if text_edition:
                                st.download_button(label="Download", data=string_total, file_name="text/.txt")

                            elif code_edition:
                                st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1]=="xlsx":
                    data = pd.read_excel(files)
                    data = pd.DataFrame(data)
                    st.dataframe(data)
                    col = st.text_input("Enter the target column for y value")
                    string_marker=string_marker+"""data = pd.read_excel(files)\n
data = pd.DataFrame(data)
col = input("Enter the target column for y value")\n"""
                    if col:
                        moded = pd.get_dummies(data)
                        X = moded.drop(col, axis=1)
                        Y = moded[col]
                        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input,
                                                                            test_size=(split_input / 100))

                        epochs_count = st.number_input("Enter the count of epochs", value=0, key=5)
                        string_marker=string_marker+"""moded = pd.get_dummies(data)\n
X = moded.drop(col, axis=1)
Y = moded[col]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input,test_size=(split_input / 100))

epochs_count = int(input("Enter the count of epochs"))\n"""


                        if epochs_count:
                            model.fit(x_train, y_train, epochs=epochs_count)
                            y_predict = model.predict(x_test)
                            string_marker=string_marker+"""history=model.fit(x_train, y_train, epochs=epochs_count)\n
                            y_predict = model.predict(x_test)\n"""
                            st.write(y_predict)


                            string_total = string_make + """\n""" + string_marker
                            text_edition = st.checkbox("Download .txt edition of the code ")
                            code_edition = st.checkbox("Download .py edition of the code ")
                            if text_edition:
                                st.download_button(label="Download", data=string_total, file_name="text/.txt")

                            elif code_edition:
                                st.download_button(label="Download", data=string_total, file_name="python/.py")





        elif selector1=="Logistic Regression":
            n_inputs4=st.number_input("Enter the number of neurons you want to add",value=0,key=6)
            n_inputs5=st.number_input("Enter the number of dense layers you want to add",value=0,key=7)
            n_inputs6=st.number_input("Enter the learning rate of neurons you want to add",value=0,key=8)
            a=[]
            for i in range(n_inputs5):
                a.append(tf.keras.layers.Dense(n_inputs4,activation="relu"))
            model1=tf.keras.Sequential([j for j in a])
            selector4=st.selectbox("Select the type of cost function",("mae","mse","binary_cross_entropy"),key=4)
            selector5=st.selectbox("Select the type of optimizer",("sgd","ADAM","RMSprop"),key=5)


            string_maker="""n_inputs4=int(input("Enter the number of neurons you want to add")\n
n_inputs5=int(input("Enter the number of dense layers you want to add"))
n_inputs6=int(input("Enter the learning rate of neurons you want to add"))
a=[]
for i in range(n_inputs5):
    a.append(tf.keras.layers.Dense(n_inputs4,activation="relu"))
model1=tf.keras.Sequential([j for j in a])\n"""

            apply1=st.checkbox("Apply custom learning rate",key=1)
            if apply1:
                if selector4=="mae":
                    if selector5=="sgd":
                        model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs6),metrics=["mae"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs6),metrics=["mae"])\n"""


                    elif selector5=="ADAM":
                        model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs6),metrics=["mae"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs6),metrics=["mae"])\n"""


                    elif selector5=="RMSprop":
                        model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs6),metrics=["mae"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs6),metrics=["mae"])\n"""


                elif selector4=="mse":
                    if selector5=="sgd":
                        model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs6),metrics=["mse"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs6),metrics=["mse"])\n"""


                    elif selector5=="ADAM":
                        model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs6),metrics=["mse"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs6),metrics=["mse"])\n"""

                    elif selector5=="RMSprop":
                        model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs6),metrics=["mse"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs6),metrics=["mse"])\n"""
        
        
                elif selector4=="binary_cross_entropy":
                    if selector5=="sgd":
                        model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs6),metrics=["accuracy"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs6),metrics=["accuracy"])\n"""


                    elif selector5=="ADAM":
                        model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs6),metrics=["accuracy"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs6),metrics=["accuracy"])\n"""

                    elif selector5=="RMSprop":
                        model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs6),metrics=["accuracy"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs6),metrics=["accuracy"])\n"""

            else:
                if selector4 == "mae":
                    if selector5 == "sgd":
                        model1.compile(loss=tf.keras.losses.mae,
                                       optimizer=tf.keras.optimizers.SGD(), metrics=["mae"])

                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(), metrics=["mae"])\n"""

                    elif selector5 == "ADAM":
                        model1.compile(loss=tf.keras.losses.mae,
                                       optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])\n"""


                    elif selector5 == "RMSprop":
                        model1.compile(loss=tf.keras.losses.mae,
                                       optimizer=tf.keras.optimizers.RMSprop(), metrics=["mae"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.RMSprop(), metrics=["mae"])\n"""


                elif selector4 == "mse":
                    if selector5 == "sgd":
                        model1.compile(loss=tf.keras.losses.mse,
                                       optimizer=tf.keras.optimizers.SGD(), metrics=["mse"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.SGD(), metrics=["mse"])\n"""


                    elif selector5 == "ADAM":
                        model1.compile(loss=tf.keras.losses.mse,
                                       optimizer=tf.keras.optimizers.Adam(), metrics=["mse"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.Adam(), metrics=["mse"])\n"""


                    elif selector5 == "RMSprop":
                        model1.compile(loss=tf.keras.losses.mse,
                                       optimizer=tf.keras.optimizers.RMSprop(), metrics=["mse"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.mse,optimizer=tf.keras.optimizers.RMSprop(), metrics=["mse"])\n"""


                elif selector4 == "binary_cross_entropy":
                    if selector5 == "sgd":
                        model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                       optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])\n"""

                    elif selector5 == "ADAM":
                        model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                       optimizer=tf.keras.optimizers.Adam(),
                                       metrics=["accuracy"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])\n"""



                    elif selector5 == "RMSprop":
                        model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                       optimizer=tf.keras.optimizers.RMSprop(),
                                       metrics=["accuracy"])
                        string_maker=string_maker+"""model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])\n"""



            rs_input1=st.number_input("Enter the random state of the model",value=0,key=9)
            split_input1=st.number_input("Enter the amount for testing samples",value=10,key=10)
            string_maker=string_maker+"""rs_input1=int(input("Enter the random state of the model"))\n
split_input1=int(input("Enter the amount for testing samples"))\n"""

            files1=st.file_uploader("upload files",type=["csv","xls","xlsx"],key=1)
            string_maker = string_maker + """files1=open()#enter the directory of your spreadsheet/dataset file\n"""
            if files1 is not None:
                if files1.name.split(".")[1] == "csv":
                    data = pd.read_csv(files1)
                    data = pd.DataFrame(data)
                    st.dataframe(data)
                    col = st.text_input("Enter the target coulmn for y value")
                    string_maker=string_maker+"""data = pd.read_csv(files1)
data = pd.DataFrame(data)
col = input("Enter the target column for y value")\n"""
                    if col:
                        moded = pd.get_dummies(data)
                        X = moded.drop(col, axis=1)
                        Y = moded[col]
                        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input1,
                                                                            test_size=(split_input1 / 100))

                        epochs_count = st.number_input("Enter the count of epochs", value=0, key=5)
                        string_maker=string_maker+"""moded = pd.get_dummies(data)
X = moded.drop(col, axis=1)
Y = moded[col]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input1,test_size=(split_input1 / 100))

epochs_count = int(input("Enter the count of epochs"))\n"""


                        if epochs_count:
                            model1.fit(x_train, y_train, epochs=epochs_count)
                            y_predict = model1.predict(x_test)
                            string_maker=string_maker+"""history=model1.fit(x_train, y_train, epochs=epochs_count)\n
y_predict = model1.predict(x_test)\n"""
                            st.write(y_predict)


                            string_total = string_make + """\n""" + string_maker
                            text_edition = st.checkbox("Download .txt edition of the code ")
                            code_edition = st.checkbox("Download .py edition of the code ")
                            if text_edition:
                                st.download_button(label="Download", data=string_total, file_name="text/.txt")

                            elif code_edition:
                                st.download_button(label="Download", data=string_total, file_name="python/.py")




                elif files1.name.split(".")[1] == "xls":
                    data = pd.read_excel(files1)
                    data = pd.DataFrame(data)
                    st.dataframe(data)
                    col = st.text_input("Enter the target column for y value")
                    string_maker=string_maker+"""data = pd.read_excel(files1)
data = pd.DataFrame(data)
col = input("Enter the target column for y value")\n"""
                    if col:
                        moded = pd.get_dummies(data)
                        X = moded.drop(col, axis=1)
                        Y = moded[col]
                        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input1,
                                                                            test_size=(split_input1 / 100))

                        epochs_count = st.number_input("Enter the count of epochs", value=0, key=5)
                        string_maker=string_maker+"""moded = pd.get_dummies(data)
X = moded.drop(col, axis=1)
Y = moded[col]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input1,test_size=(split_input1 / 100))

epochs_count = int(input("Enter the count of epochs"))\n"""


                        if epochs_count:
                            model1.fit(x_train, y_train, epochs=epochs_count)
                            y_predict = model1.predict(x_test)
                            string_maker=string_maker+"""history=model1.fit(x_train, y_train, epochs=epochs_count)\n
y_predict = model1.predict(x_test)\n"""
                            st.write(y_predict)


                            string_total = string_make + """\n""" + string_maker
                            text_edition = st.checkbox("Download .txt edition of the code ")
                            code_edition = st.checkbox("Download .py edition of the code ")
                            if text_edition:
                                st.download_button(label="Download", data=string_total, file_name="text/.txt")

                            elif code_edition:
                                st.download_button(label="Download", data=string_total, file_name="python/.py")



                elif files1.name.split(".")[1] == "xlsx":
                    data = pd.read_excel(files1)
                    data = pd.DataFrame(data)
                    st.dataframe(data)
                    col = st.text_input("Enter the target column for y value")
                    string_maker=string_maker+"""data = pd.read_excel(files1)
data = pd.DataFrame(data)
col = input("Enter the target column for y value")\n"""

                    if col:
                        moded = pd.get_dummies(data)
                        X = moded.drop(col, axis=1)
                        Y = moded[col]
                        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input1,
                                                                            test_size=(split_input1 / 100))

                        epochs_count = st.number_input("Enter the count of epochs", value=0, key=5)
                        string_maker=string_maker+""" moded = pd.get_dummies(data)
X = moded.drop(col, axis=1)
Y = moded[col]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=rs_input1,test_size=(split_input1 / 100))

epochs_count = int(input("Enter the count of epochs"))\n"""


                        if epochs_count:
                            model1.fit(x_train, y_train, epochs=epochs_count)
                            y_predict = model1.predict(x_test)
                            string_maker=string_maker+"""history=model1.fit(x_train, y_train, epochs=epochs_count)\n
y_predict = model1.predict(x_test)\n"""

                            st.write(y_predict)

                            string_total = string_make + """\n""" + string_maker
                            text_edition = st.checkbox("Download .txt edition of the code ")
                            code_edition = st.checkbox("Download .py edition of the code ")
                            if text_edition:
                                st.download_button(label="Download", data=string_total, file_name="text/.txt")

                            elif code_edition:
                                st.download_button(label="Download", data=string_total, file_name="python/.py")





    elif selector=="Sklearn":
        selector6=st.selectbox("Choose Regressors",("Linear Regression","Logistic Regression","Ridge Regression","Lasso Regression"),key=6)
        files2 = st.file_uploader("upload files", type=["csv", "xls", "xlsx"], key=2)
        rs_input2 = st.number_input("Enter the random state of the model", value=0, key=14)
        split_input2 = st.number_input("Enter the amount for testing samples", value=0, key=15)
        string_maker="""files2=files1=open()#enter the directory of your spreadsheet/dataset file\n
rs_input2 = int(input("Enter the random state of the model"))\n
split_input2 = int(input("Enter the amount for testing samples"))\n"""

        if selector6=="Linear Regression":
            string_make="""import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression"""
            if files2 is not None:
                if files2.name.split(".")[1]=="csv":
                    data2 = pd.read_csv(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_csv(files2)
data2 = pd.DataFrame(data2)
st.dataframe(data2)
col2 = input("Enter the target column for y value")\n"""
                    if col2:
                        lr=LinearRegression()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                    test_size=(split_input2 / 100))
                        lr.fit(x_train,y_train)
                        string_maker=string_maker+"""lr=LinearRegression()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
lr.fit(x_train,y_train)\n"""


                        pred=st.checkbox("Want to see predictions ")
                        y_pred = lr.predict(x_test)
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = lr.predict(x_test)\n"""


                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")



                elif files2.name.split(".")[1]=="xls":
                    data2 = pd.read_excel(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker = string_maker + """data2 = pd.read_excel(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""
                    if col2:
                        lr = LinearRegression()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        lr.fit(x_train, y_train)
                        y_pred = lr.predict(x_test)
                        string_maker=string_maker+"""lr = LinearRegression()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
lr.fit(x_train, y_train)\n"""



                        pred=st.checkbox("Want to see predictions ")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = lr.predict(x_test)\n"""


                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")



                elif files2.name.split(".")[1]=="xlsx":
                    data2 = pd.read_excel(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_excel(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""
                    if col2:
                        lr = LinearRegression()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        lr.fit(x_train, y_train)
                        y_pred = lr.predict(x_test)

                        string_maker=string_maker+"""lr = LinearRegression()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
lr.fit(x_train, y_train)\n"""


                        pred=st.checkbox("Want to see predictions")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = lr.predict(x_test)\n"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")




        elif selector6=="Logistic Regression":
            string_make="""import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression"""
            if files2 is not None:
                if files2.name.split(".")[1] == "csv":
                    data2 = pd.read_csv(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target coulmn for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_csv(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""
                    if col2:
                        lo = LogisticRegression()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1,Y1, random_state=rs_input2,
                                                                                test_size=(split_input2 / 100))
                        lo.fit(x_train, y_train)

                        string_maker=string_maker+"""lo = LogisticRegression()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1,Y1, random_state=rs_input2,test_size=(split_input2 / 100))
lo.fit(x_train, y_train)\n"""

                        y_pred = lo.predict(x_test)
                        pred = st.checkbox("Want to see predictions ? ", key=0)
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = lo.predict(x_test)\n"""


                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")



                elif files2.name.split(".")[1] == "xls":
                    data2 = pd.read_excel(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_excel(files2)
data2 = pd.DataFrame(data2)
col2 =input("Enter the target column for y value")\n"""
                    if col2:
                        lo = LogisticRegression()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        lo.fit(x_train, y_train)
                        y_pred = lo.predict(x_test)
                        string_maker=string_maker+""" lo = LogisticRegression()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
lo.fit(x_train, y_train)\n"""


                        pred=st.checkbox("Want to see predictions ")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = lo.predict(x_test)"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files2.name.split(".")[1] == "xlsx":
                    data2 = pd.read_excel(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)

                    string_maker=string_maker+"""data2 = pd.read_excel(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""

                    if col2:
                        lo = LogisticRegression()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        lo.fit(x_train, y_train)
                        y_pred = lo.predict(x_test)
                        string_maker=string_maker+"""lo = LogisticRegression()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
lo.fit(x_train, y_train)\n"""


                        pred=st.checkbox("Want to see predictions ")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = lo.predict(x_test)\n"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")

        elif selector6=="Ridge Regression":
            string_make="""import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge"""
            if files2 is not None:
                if files2.name.split(".")[1] == "csv":
                    data2 = pd.read_csv(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_csv(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""
                    if col2:
                        rd = Ridge()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        rd.fit(x_train, y_train)
                        string_maker=string_maker+"""rd = Ridge()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
rd.fit(x_train, y_train)\n"""


                        y_pred = rd.predict(x_test)
                        pred=st.checkbox("Want to see predictions ")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = rd.predict(x_test)\n"""


                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files2.name.split(".")[1] == "xls":
                    data2 = pd.read_excel(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_excel(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""


                    if col2:
                        rd = Ridge()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
                        rd.fit(x_train, y_train)
                        y_pred = rd.predict(x_test)
                        string_maker=string_maker+"""rd = Ridge()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
rd.fit(x_train, y_train)\n"""


                        pred=st.checkbox("Want to see predictions ")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = rd.predict(x_test)\n"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files2.name.split(".")[1] == "xlsx":
                    data2 = pd.read_excel(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_excel(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""
                    if col2:
                        rd = Ridge()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        rd.fit(x_train, y_train)
                        string_maker=string_maker+"""rd = Ridge()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
rd.fit(x_train, y_train)"""
                        y_pred = rd.predict(x_test)
                        pred=st.checkbox("Want to see predictions ")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = rd.predict(x_test)\n"""


                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")

        elif selector6=="Lasso Regression":
            string_make="""import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso"""
            if files2 is not None:
                if files2.name.split(".")[1] == "csv":
                    data2 = pd.read_csv(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_csv(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""


                    if col2:
                        las = Lasso()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        las.fit(x_train, y_train)
                        y_pred = las.predict(x_test)
                        string_maker=string_maker+"""las = Lasso()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
las.fit(x_train, y_train)\n"""

                        pred=st.checkbox("Want to see predictions")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = las.predict(x_test)\n"""


                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files2.name.split(".")[1] == "xls":
                    data2 = pd.read_excel(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target column for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_excel(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target coulmn for y value")\n"""


                    if col2:
                        las = Lasso()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        las.fit(x_train, y_train)

                        y_pred = las.predict(x_test)
                        string_maker=string_maker+"""las = Lasso()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
las.fit(x_train, y_train)\n"""

                        pred=st.checkbox("Want to see predictions ")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = las.predict(x_test)\n"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files2.name.split(".")[1] == "xlsx":
                    data2 = pd.read_excel(files2)
                    data2 = pd.DataFrame(data2)
                    st.dataframe(data2)
                    col2 = st.text_input("Enter the target coulmn for y value", key=3)
                    string_maker=string_maker+"""data2 = pd.read_excel(files2)
data2 = pd.DataFrame(data2)
col2 = input("Enter the target column for y value")\n"""

                    if col2:
                        las = Lasso()
                        moded2 = pd.get_dummies(data2)
                        X1 = moded2.drop(col2, axis=1)
                        Y1 = moded2[col2]
                        x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,
                                                                            test_size=(split_input2 / 100))
                        las.fit(x_train, y_train)
                        string_maker=string_maker+"""las = Lasso()
moded2 = pd.get_dummies(data2)
X1 = moded2.drop(col2, axis=1)
Y1 = moded2[col2]
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, random_state=rs_input2,test_size=(split_input2 / 100))
las.fit(x_train, y_train)\n"""


                        y_pred = las.predict(x_test)
                        pred=st.checkbox("Want to see prediction ")
                        if pred:
                            st.write(y_pred)
                            string_maker=string_maker+"""y_pred = las.predict(x_test)\n"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")






    




