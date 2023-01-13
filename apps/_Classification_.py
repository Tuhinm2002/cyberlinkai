import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import  Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report,plot_confusion_matrix
import cv2 as cv
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
    st.write("# Classification")

    selector=st.selectbox("Choose Machine Learning Library",("Sklearn","Tensorflow"),key=0)
    if selector=="Tensorflow":
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        selector1=st.selectbox("Choose Classifier",("Multi-class","Binary"),key=1)

        n_inputs=st.number_input("Enter the number of neurons",value=0,key=0)
        n_inputs1 = st.number_input("Enter the number of Dense layers", value=0, key=1)

        n_inputs3=st.number_input("Enter the input size",key=3,value=0)
        slide_input=st.slider("Choose striding limit",key=0)
        slide_input1 = st.slider("Choose color type like 1 for binary and 3 for RGB", key=1,max_value=5,min_value=1)
        string_make="""import numpy as np
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from PIL import  Image
import cv2 as cv
from sklearn.model_selection import train_test_split
from keras.preprocessing import image #Optional...! use this when you use image classification"""
        string_maker="""n_inputs=int(input("Enter the number of neurons"))
n_inputs1 = int(input("Enter the number of Dense layers"))

n_inputs3=int(input("Enter the input size"))
slide_input=int(input("Choose striding limit"))
slide_input1 = int(input("Choose color type like 1 for binary and 3 for RGB"))\n"""



        if n_inputs3 and slide_input:
            model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(n_inputs,(slide_input,slide_input),activation="relu",input_shape=(n_inputs3,n_inputs3,slide_input1))])
            string_maker=string_maker+"""model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(n_inputs,(slide_input,slide_input),activation="relu",input_shape=(n_inputs3,n_inputs3,slide_input1))])\n"""

            check=st.checkbox("Select this to apply dynamic neural networks and unselect for static",key=0)
            if check:
                for i in range(n_inputs1-1):
                    model.add(tf.keras.layers.Conv2D(n_inputs*2,(slide_input,slide_input),activation="relu"))
                    model.add(tf.keras.layers.MaxPooling2D(slide_input-1,slide_input-1))

                string_maker=string_maker+"""for i in range(n_inputs1-1):
\tmodel.add(tf.keras.layers.Conv2D(n_inputs*2,(slide_input,slide_input),activation="relu"))
\tmodel.add(tf.keras.layers.MaxPooling2D(slide_input-1,slide_input-1))\n"""
            else:
                for i in range(n_inputs1-1):
                    model.add(tf.keras.layers.Conv2D(n_inputs,(slide_input,slide_input),activation="relu"))
                    model.add(tf.keras.layers.MaxPooling2D(slide_input-1,slide_input-1))

                string_maker=string_maker+"""for i in range(n_inputs1-1):
\tmodel.add(tf.keras.layers.Conv2D(n_inputs,(slide_input,slide_input),activation="relu"))
\tmodel.add(tf.keras.layers.MaxPooling2D(slide_input-1,slide_input-1))\n"""

            if selector1=="Binary":
                model.add(tf.keras.layers.Dense(n_inputs*n_inputs1,activation="relu"))
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

                string_maker=string_maker+"""model.add(tf.keras.layers.Dense(n_inputs*n_inputs1,activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))\n"""

                selector2=st.selectbox("Choose loss function",("binary_cross_entropy","categorical_cross_entropy","sparse_categorical_cross_entropy"))
                selector3 = st.selectbox("Choose optimizers", ("SGD", "Adam", "RmsProp"), key=3)
                apply=st.checkbox("Apply custom learning rate",key=1)
            # if n_inputs3 and slide_input:
                if apply:
                    n_inputs2 = st.number_input("Enter the learning rate of neurons", key=2)
                    string_maker=string_maker+"""n_inputs2 =int(input("Enter the learning rate of neurons"))\n"""
                    if selector2=="binary_cross_entropy":
                        if selector3=="Adam":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])\n"""
                        elif selector3=="SGD":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

                        elif selector3=="RmsProp":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])\n"""



                    elif selector2=="categorical_cross_entropy":
                        if selector3=="Adam":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

                        elif selector3=="SGD":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])\n"""


                        elif selector3=="RmsProp":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

                    elif selector2=="sparse_categorical_cross_entropy":
                        if selector3=="Adam":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

                        elif selector3=="SGD":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                      optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2), metrics=["accuracy"])\n"""


                        elif selector3=="RmsProp":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                      optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2), metrics=["accuracy"])\n"""


                else:
                    if selector2 == "binary_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])\n"""


                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])\n"""



                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer=tf.keras.optimizers.RMSprop(),
                                      metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])\n"""

                    elif selector2 == "categorical_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                      optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])\n"""


                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                      optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])\n"""


                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                      optimizer=tf.keras.optimizers.RMSprop(),
                                      metrics=["accuracy"])

                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])\n"""


                    elif selector2 == "sparse_categorical_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                      optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])\n"""


                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                      optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])\n"""


                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                      optimizer=tf.keras.optimizers.RMSprop(),
                                      metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])\n"""


                rs_input=st.number_input("Enter the random state of the model",value=0,key=4)
                split_input=st.number_input("Enter the amount fo testing samples",value=0,key=5)
                string_maker=string_maker+"""rs_input=int(input("Enter the random state of the model"))\n
split_input=int(input("Enter the amount fo testing samples"))\n"""


                selector4=st.selectbox("Choose classification type",("Image","DataFrame"),key=4)
                if selector4=="Image":
                    files=st.file_uploader("Upload first item",type=["png","jpg","jpeg"],key=0,accept_multiple_files=True)
                    name=st.text_input("Name of the first item",key=0)
                    # files2=st.file_uploader()
                    train_data = []
                    files1=st.file_uploader("Upload second item",type=["png","jpg","jpeg"],key=1,accept_multiple_files=True)
                    name1=st.text_input("Name of the second item",key=1)
                    st.write("Enter the prediction data")
                    files2 = st.file_uploader("Upload files", type=["jpg", "png", "jpeg"], key=1,accept_multiple_files=True)
                    string_maker=string_maker+"""files=open() # specify the path of your first item
name=input("Name of the first item")
train_data = []
files1=open() #specify the path of your second item
name1=input("Name of the second item")\n"""

                    name_appender=[]
                    name_appender.append(name)
                    name_appender.append(name1)
                    string_maker=string_maker+"""name_appender=[]
name_appender.append(name)
name_appender.append(name1)\n"""

                    if files is not None and files1 is not None and name1 is not None and name is not None and rs_input !=0 and split_input !=0:
                        for i in files:
                            img=Image.open(i)
                            img_array=np.array(img)
                            img_new=cv.resize(img_array,(n_inputs3,n_inputs3))
                            train_data.append([img_new,name_appender.index(name)])
                        for j in files1:
                            img1=Image.open(j)
                            img1_array=np.array(img1)
                            img_new1=cv.resize(img1_array,(n_inputs3,n_inputs3))
                            train_data.append([img_new1,name_appender.index(name1)])
                        for file in files2:
                            img2=Image.open(file)
                            x_array=np.array(img2)
                            img_array=cv.resize(x_array,(n_inputs3,n_inputs3))
                            x = np.expand_dims(img_array, axis=0)
                            images = np.vstack([x])
                        X = []
                        Y = []
                        for features,labels in train_data:
                            X.append(features)
                            Y.append(labels)
                        X=np.array(X).reshape(-1,n_inputs3,n_inputs3,slide_input1)
                        Y=np.array(Y)


                        x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=int(rs_input),test_size=split_input/100)


                        batch=st.checkbox("Want to keep batch size same as random state...",key=2)

                        string_maker=string_maker+"""for i in files:
\timg=Image.open(i)
\timg_array=np.array(img)
\timg_new=cv.resize(img_array,(n_inputs3,n_inputs3))
\ttrain_data.append([img_new,name_appender.index(name)])
for j in files1:
    img1=Image.open(j)
    img1_array=np.array(img1)
    img_new1=cv.resize(img1_array,(n_inputs3,n_inputs3))
    train_data.append([img_new1,name_appender.index(name1)])
X = []
Y = []
for features,labels in train_data:
    X.append(features)
    Y.append(labels)
X=np.array(X).reshape(-1,n_inputs3,n_inputs3,slide_input1)
Y=np.array(Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=int(rs_input),test_size=split_input/100)\n"""
                        string_maker = string_maker + """files2=open() # Specify the path for predicting data
for file in files2:
    img2=Image.open(file)
    x_array=np.array(img2)
    img_array=cv.resize(x_array,(n_inputs3,n_inputs3))
    x = np.expand_dims(img_array, axis=0)
    images = np.vstack([x])\n"""


                        if batch:
                            epoch=st.number_input("Enter the number of epochs ",value=0,key=6)
                            # validation=st.number_input("Enter the number of validation steps",value=0,key=7)
                            batch_num1=rs_input
                            if epoch !=0:
                                model.fit(x_train,y_train,batch_size=batch_num1,epochs=epoch)
                                classes = model.predict(images, batch_size=batch_num1)
                                if classes[0] > 0:
                                    st.write("is a", name)
                                    string_maker = string_maker + """epoch=int(input("Enter the number of epochs "))
validation=int(input("Enter the number of validation steps"))
batch_num1=rs_input
history=model.fit(x_train,y_train,batch_size=batch_num1,validation_data=(x_test,y_test),epochs=epoch,validation_steps=validation)
classes=model.predict(images,batch_size=batch_num1)\n
    if classes[0]>0:
        print("is a",name)
    else:
        print("is a",name1)\n"""
                                    string_total = string_make + """\n""" + string_maker
                                    text_edition = st.checkbox("Download .txt edition of the code ")
                                    code_edition = st.checkbox("Download .py edition of the code ")
                                    if text_edition:
                                        st.download_button(label="Download", data=string_total, file_name="text/.tx")

                                    elif code_edition:
                                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                                else:
                                    st.write("is a", name1)
                                    string_maker=string_maker+"""epoch=int(input("Enter the number of epochs "))
validation=int(input("Enter the number of validation steps"))
batch_num1=rs_input
history=model.fit(x_train,y_train,batch_size=batch_num1,validation_data=(x_test,y_test),epochs=epoch,validation_steps=validation)\n
classes=model.predict(images,batch_size=batch_num1)\n
    if classes[0]>0:
        print("is a",name)
    else:
        print("is a",name1)\n"""
                                    string_total = string_make + """\n""" + string_maker
                                    text_edition = st.checkbox("Download .txt edition of the code ")
                                    code_edition = st.checkbox("Download .py edition of the code ")
                                    if text_edition:
                                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                    elif code_edition:
                                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                        else:
                            batch_num=st.number_input("Enter the amount of batch size ",value=1,key=8)
                            epoch = st.number_input("Enter the number of epochs ", value=0, key=6)
                            # validation = st.number_input("Enter the number of validation steps", value=0, key=7)
                            batch_num1=batch_num
                            if epoch !=0 :
                                model.fit(x_train, y_train, batch_size=batch_num1, epochs=epoch)
                                classes = model.predict(images, batch_size=batch_num1)
                                if classes[0] > 0:
                                    st.write("is a", name)
                                    string_maker = string_maker + """batch_num=int(input("Enter the amount of batch size "))
epoch = int(input("Enter the number of epochs "))
validation = int(input("Enter the number of validation steps"))
batch_num1=batch_num
history=model.fit(x_train, y_train, batch_size=batch_num1, validation_data=(x_test, y_test), epochs=epoch,validation_steps=validation)\n
classes=model.predict(images,batch_size=batch_num1)\n
    if classes[0]>0:
        print("is a",name)
    else:
        print("is a",name1)\n"""
                                    string_total = string_make + """\n""" + string_maker
                                    text_edition = st.checkbox("Download .txt edition of the code ")
                                    code_edition = st.checkbox("Download .py edition of the code ")
                                    if text_edition:
                                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                    elif code_edition:
                                        st.download_button(label="Download", data=string_total, file_name="python/.py")
                                else:
                                    st.write("is a", name1)
                                    string_maker = string_maker + """batch_num=int(input("Enter the amount of batch size "))
epoch = int(input("Enter the number of epochs "))
validation = int(input("Enter the number of validation steps"))
batch_num1=batch_num
history=model.fit(x_train, y_train, batch_size=batch_num1, validation_data=(x_test, y_test), epochs=epoch,validation_steps=validation)\n
classes=model.predict(images,batch_size=batch_num1)\n
    if classes[0]>0:
        print("is a",name)
    else:
        print("is a",name1)\n"""
                                    string_total = string_make + """\n""" + string_maker
                                    text_edition = st.checkbox("Download .txt edition of the code ")
                                    code_edition = st.checkbox("Download .py edition of the code ")
                                    if text_edition:
                                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                    elif code_edition:
                                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                            


                    elif selector4=="DataFrame":
                        files = st.file_uploader("Upload first item", type=["csv", "xls", "xlsx"], key=0,accept_multiple_files=True)
                        str1 = st.text_input("Enter the target column", key=1)
                        string_maker=string_maker+"""files=() # Specify the path for your data \n"""
                        string_maker=string_maker+"""str1 = input("Enter the target column")"""
                        if files is not None:
                            if files.name.split(".")[1]=="csv":
                                data=pd.read_csv(files)
                                data=pd.get_dummies(data)
                                x=data.drop(str1,axis=1)
                                y=data[str1]
                                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=split_input / 100)
                                epoch = st.number_input("Enter the number of epochs ", value=1, key=6)
                                model.fit(x_train, y_train,epochs=epoch)


                                string_maker=string_maker+"""data=pd.read_csv(files)
data=pd.get_dummies(data)
x=data.drop(str1,axis=1)
y=data[str1]
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=split_input / 100)
epoch = int(input("Enter the number of epochs "))
history = model.fit(x_train, y_train,epochs=epoch)"""





                                prediction = model.predict(x_test)
                                st.write(prediction)
                                string_maker=string_maker+"""prediction = model.predict(x_test)"""


                                string_total = string_make + """\n""" + string_maker
                                text_edition = st.checkbox("Download .txt edition of the code ")
                                code_edition = st.checkbox("Download .py edition of the code ")
                                if text_edition:
                                    st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                elif code_edition:
                                    st.download_button(label="Download", data=string_total, file_name="python/.py")




                            elif files.name.split(".")[1] == "xls":
                                data = pd.read_excel(files)
                                data=pd.get_dummies(data)
                                x=data.drop(str1,axis=1)
                                y=data[str1]
                                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=split_input / 100)
                                epoch = st.number_input("Enter the number of epochs ", value=1, key=6)
                                model.fit(x_train, y_train,epochs=epoch)
                                string_maker=string_maker+"""data = pd.read_excel(files)
data=pd.get_dummies(data)
x=data.drop(str1,axis=1)
y=data[str1]
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=split_input / 100)
epoch = int(input("Enter the number of epochs "))
history = model.fit(x_train, y_train,epochs=epoch)\n"""





                                prediction = model.predict(x_test)
                                st.write(prediction)
                                string_maker=string_maker+"""prediction = model.predict(x_test)\n"""


                                string_total = string_make + """\n""" + string_maker
                                text_edition = st.checkbox("Download .txt edition of the code ")
                                code_edition = st.checkbox("Download .py edition of the code ")
                                if text_edition:
                                    st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                elif code_edition:
                                    st.download_button(label="Download", data=string_total, file_name="python/.py")






                            elif files.name.split(".")[1] == "xlsx":
                                data = pd.read_excel(files)
                                data=pd.get_dummies(data)
                                x=data.drop(str1,axis=1)
                                y=data[str1]
                                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),
                                                                                    test_size=split_input / 100)

                                epoch = st.number_input("Enter the number of epochs ", value=1, key=6)
                                model.fit(x_train, y_train,epochs=epoch)
                                string_maker=string_maker+"""data = pd.read_excel(files)
data=pd.get_dummies(data)
x=data.drop(str1,axis=1)
y=data[str1]
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=split_input / 100)

epoch = int(input("Enter the number of epochs "))
history = model.fit(x_train, y_train,epochs=epoch)\n"""





                                prediction=model.predict(x_test)
                                st.write(prediction)
                                string_maker=string_maker+"""prediction=model.predict(x_test)\n"""


                                string_total = string_make + """\n""" + string_maker
                                text_edition = st.checkbox("Download .txt edition of the code ")
                                code_edition = st.checkbox("Download .py edition of the code ")
                                if text_edition:
                                    st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                elif code_edition:
                                    st.download_button(label="Download", data=string_total, file_name="python/.py")




            elif selector1=="Multi-class":
                model.add(tf.keras.layers.Dense(n_inputs * n_inputs1, activation="relu"))
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(1, activation="softmax"))

                string_maker=string_maker+"""model.add(tf.keras.layers.Dense(n_inputs * n_inputs1, activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation="softmax"))\n"""

                selector2 = st.selectbox("Choose loss function", (
                "binary_cross_entropy", "categorical_cross_entropy", "sparse_categorical_cross_entropy"))
                selector3 = st.selectbox("Choose optimizers", ("SGD", "Adam", "RmsProp"), key=3)
                apply = st.checkbox("Apply custom learning rate", key=10)
                # if n_inputs3 and slide_input:
                if apply:
                    n_inputs2 = st.number_input("Enter the learning rate of neurons", key=2)
                    if selector2 == "binary_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                          optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                          optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])\n"""


                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                          optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),
                                          metrics=["accuracy"])

                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])\n"""

                    elif selector2 == "categorical_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),
                                          metrics=["accuracy"])

                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])\n"""


                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])\n"""


                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])\n"""


                    elif selector2 == "sparse_categorical_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=n_inputs2),metrics=["accuracy"])\n"""


                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=n_inputs2),metrics=["accuracy"])\n"""


                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(learning_rate=n_inputs2),metrics=["accuracy"])\n"""


                else:
                    if selector2 == "binary_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                          optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])\n"""


                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                          optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])\n"""

                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                          optimizer=tf.keras.optimizers.RMSprop(),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])\n"""


                    elif selector2 == "categorical_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])\n"""


                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])\n"""


                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.RMSprop(),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])\n"""


                    elif selector2 == "sparse_categorical_cross_entropy":
                        if selector3 == "Adam":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])\n"""

                        elif selector3 == "SGD":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])\n"""


                        elif selector3 == "RmsProp":
                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                          optimizer=tf.keras.optimizers.RMSprop(),
                                          metrics=["accuracy"])
                            string_maker=string_maker+"""model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(),metrics=["accuracy"])\n"""


                rs_input = st.number_input("Enter the random state of the model", value=0, key=4)
                split_input = st.number_input("Enter the amount fo testing samples", value=0, key=5)
                string_maker=string_maker+"""rs_input = int(input("Enter the random state of the model"))\n
split_input =int(input("Enter the amount fo testing samples"))\n"""

                train_data = []
                values=st.number_input("Enter the number of classes")
                name_appender = []
                string_maker=string_maker+""" train_data = []\n
values=int(input("Enter the number of classes"))\n
name_appender = []\n"""
                val_count=0
                for val in range(int(values)):
                    files1 = st.file_uploader("Upload second item", type=["png", "jpg", "jpeg"], key=val,accept_multiple_files=True)
                    val_count+=1
                    name1 = st.text_input("Name of the item",key=val) #str(val)
                    name_appender.append(name1)
                    if files1 is not None and name1 is not None and rs_input !=0 and split_input !=0:
                        for i in files1:
                            img = Image.open(i)
                            img_array = np.array(img)
                            img_new = cv.resize(img_array, (n_inputs3, n_inputs3))
                            train_data.append([img_new, name_appender.index(name1)])
                files2 = st.file_uploader("Upload files", type=["jpg", "png", "jpeg"], key=val_count+1,accept_multiple_files=True)
                for file in files2:
                    img2 = Image.open(file)
                    x_array = np.array(img2)
                    img_array = cv.resize(x_array, (n_inputs3, n_inputs3))
                    x = np.expand_dims(img_array, axis=0)
                    images = np.vstack([x])

                string_maker=string_maker+"""files1 = open() #Specify the path of your data
name_appender = []
for val in range(values):
    name1 = input("Name of the", val, "th  item")
    name_appender.append(name1)
for i in files1: #use os.listdir for iterating through your directory
    img = Image.open(i)
    img_array = np.array(img)
    img_new = cv.resize(img_array, (n_inputs3, n_inputs3))
    train_data.append([img_new, name_appender.index(name1)])\n"""
                string_maker = string_maker + """files2 = st.file_uploader("Upload files", type=["jpg", "png", "jpeg"], key=1,accept_multiple_files=True)
ab=0
for file in files2:
    img2 = Image.open(file)
    x_array = np.array(img2)
    img_array = cv.resize(x_array, (n_inputs3, n_inputs3))
    x = np.expand_dims(img_array, axis=0)
    images = np.vstack([x])"""




                if len(train_data)!=0:
                    X = []
                    Y = []
                    for features, labels in train_data:
                        X.append(features)
                        Y.append(labels)
                    X = np.array(X).reshape(-1, n_inputs3, n_inputs3, slide_input1)
                    Y=np.array(Y)

                    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=int(rs_input),test_size=int(split_input) / 100)
                    ab = 0

                    string_maker=string_maker+"""X = []
Y = []
for features, labels in train_data:
    X.append(features)
    Y.append(labels)
X = np.array(X).reshape(-1, n_inputs3, n_inputs3, slide_input1)
Y=np.array(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=int(rs_input),test_size=split_input / 100)\n"""

                    batch = st.checkbox("Want to keep batch size same as random state...",key=0)
                    if batch:
                        epoch = st.number_input("Enter the number of epochs ", value=0, key=6)
                        batch_num1 = rs_input
                        if epoch != 0 :
                            model.fit(x_train, y_train, batch_size=batch_num1, epochs=epoch)
                            classes = model.predict(images, batch_size=batch_num1)
                            while ab < len(name_appender):
                                if classes[0] > name_appender.index(name_appender[0]):
                                    st.write("is a", name_appender[0])
                                    string_maker = string_maker + """epoch = int(input("Enter the number of epochs "))
validation = int(input("Enter the number of validation steps"))
batch_num1 = rs_input
history=model.fit(x_train, y_train, batch_size=batch_num1, validation_data=(x_test, y_test), epochs=epoch,validation_steps=validation)\n
classes = model.predict(images, batch_size=batch_num1)
    while ab < range(len(name_appender)):
        if classes[0]>name_appender.index(name_appender[0]):
            st.write("is a", name_appender[0])
            break
        else:
            st.write("is a", name_appender[ab])
            break\n"""
                                    string_total = string_make + """\n""" + string_maker
                                    text_edition = st.checkbox("Download .txt edition of the code ")
                                    code_edition = st.checkbox("Download .py edition of the code ")
                                    if text_edition:
                                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                    elif code_edition:
                                        st.download_button(label="Download", data=string_total, file_name="python/.py")
                                    break
                                else:
                                    st.write("is a", name_appender[ab])
                                    string_maker = string_maker + """epoch = int(input("Enter the number of epochs "))
validation = int(input("Enter the number of validation steps"))
batch_num1 = rs_input
history=model.fit(x_train, y_train, batch_size=batch_num1, validation_data=(x_test, y_test), epochs=epoch,validation_steps=validation)\n
classes = model.predict(images, batch_size=batch_num1)
    while ab < range(len(name_appender)):
        if classes[0]>name_appender.index(name_appender[0]):
            st.write("is a", name_appender[0])
            break
        else:
            st.write("is a", name_appender[ab])
            break\n"""
                                    string_total = string_make + """\n""" + string_maker
                                    text_edition = st.checkbox("Download .txt edition of the code ")
                                    code_edition = st.checkbox("Download .py edition of the code ")
                                    if text_edition:
                                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                    elif code_edition:
                                        st.download_button(label="Download", data=string_total, file_name="python/.py")
                                    break

                    else:
                        batch_num = st.number_input("Enter the amount of batch size ", value=1, key=8)
                        epoch = st.number_input("Enter the number of epochs ", value=0, key=6)
                        batch_num1 = batch_num
                        if epoch != 0 :
                            model.fit(x_train, y_train, batch_size=batch_num1,epochs=epoch)
                            classes = model.predict(images, batch_size=batch_num1)
                            while ab < len(name_appender):
                                if classes[0] > name_appender.index(name_appender[0]):
                                    st.write("is a", name_appender[0])
                                    string_maker = string_maker + """batch_num = int(input("Enter the amount of batch size "))
epoch = int(input("Enter the number of epochs "))
validation = int(input("Enter the number of validation steps"))
batch_num1 = batch_num
history=model.fit(x_train, y_train, batch_size=batch_num1, validation_data=(x_test, y_test),epochs=epoch,validation_steps=validation)\n
classes = model.predict(images, batch_size=batch_num1)
    while ab < range(len(name_appender)):
        if classes[0]>name_appender.index(name_appender[0]):
            st.write("is a", name_appender[0])
            break
        else:
            st.write("is a", name_appender[ab])
            break\n"""
                                    string_total = string_make + """\n""" + string_maker
                                    text_edition = st.checkbox("Download .txt edition of the code ")
                                    code_edition = st.checkbox("Download .py edition of the code ")
                                    if text_edition:
                                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                    elif code_edition:
                                        st.download_button(label="Download", data=string_total, file_name="python/.py")
                                    break
                                else:
                                    st.write("is a", name_appender[ab])
                                    string_maker = string_maker + """batch_num = int(input("Enter the amount of batch size "))
epoch = int(input("Enter the number of epochs "))
validation = int(input("Enter the number of validation steps"))
batch_num1 = batch_num
history=model.fit(x_train, y_train, batch_size=batch_num1, validation_data=(x_test, y_test),epochs=epoch,validation_steps=validation)\n
classes = model.predict(images, batch_size=batch_num1)
    while ab < range(len(name_appender)):
        if classes[0]>name_appender.index(name_appender[0]):
            st.write("is a", name_appender[0])
            break
        else:
            st.write("is a", name_appender[ab])
            break\n"""
                                    string_total = string_make + """\n""" + string_maker
                                    text_edition = st.checkbox("Download .txt edition of the code ")
                                    code_edition = st.checkbox("Download .py edition of the code ")
                                    if text_edition:
                                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                    elif code_edition:
                                        st.download_button(label="Download", data=string_total, file_name="python/.py")
                                    break

    elif selector=="Sklearn":
        s=st.selectbox("Choose classifier",("Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",),key=0)
        if s=="Nearest Neighbors":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix"""
            n=st.number_input("Enter the number neighbours",key=0)

            files=st.file_uploader("Choose files",type=["xls","xlsx","csv"])
            string_maker="""n=int(input("Enter the number neighbours"))\n

files=open() # Specify the path of the data \n"""


            if files is not None:
                str1 = st.text_input("Enter the target column ")
                string_maker=string_maker+"""str1 = input("Enter the target column ")"""
                if files.name.split(".")[1]=="csv":
                    reader=pd.read_csv(files)
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf=KNeighborsClassifier(int(n))
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input=st.number_input("Enter the train test split size ", key=2)
                    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train,y_train)
                    string_maker=string_maker+"""reader=pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf=KNeighborsClassifier(n)
rs_input = int(input("Enter the random state"))
ts_input=int(input("Enter the train test split size "))
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train,y_train)\n"""

                    prediction = clf.predict(x_test)
                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = KNeighborsClassifier(n)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_excel(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = KNeighborsClassifier(n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = KNeighborsClassifier(n)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = KNeighborsClassifier(n)
rs_input = st.number_input("Enter the random state", key=1)
ts_input = st.number_input("Enter the train test split size ", key=2)
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")



        elif s == "Linear SVM":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix"""
            n = st.number_input("Enter the Regularization value", key=0)

            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            str1=st.text_input("Enter the target column ")
            string_maker="""n = int(input("Enter the Regularization value"))\n
files = open() #Specify the path of the data\n
str1=input("Enter the target column ")\n"""

            if files is not None and n!=0:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = SVC(kernel="linear",C=n)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),
                                                                        test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(kernel="linear",C=n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = SVC(kernel="linear",C=n)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),
                                                                        test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_excel(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(kernel="linear",C=n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")



                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = SVC(kernel="linear",C=n)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(kernel="linear",C=n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(reader, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")



        elif s == "RBF SVM":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,plot_confusion_matrix"""
            n = st.number_input("Enter the Regularization value", key=0)
            n1=st.number_input("Enter the gamma value ",key=3)
            string_maker="""n = int(input("Enter the Regularization value"))\n
n1=int(input("Enter the gamma value "))\n"""
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"],key=0)
            str1=st.text_input("Enter the target column")
            string_maker=string_maker+"""files = open() #Specify the path of the data\n
str1=input("Enter the target column")\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)\n"""
                    if n != 0 and n1 != 0:
                        clf = SVC(gamma=n1, C=n)
                        rs_input = st.number_input("Enter the random state", key=1)
                        ts_input = st.number_input("Enter the train test split size ", key=2)
                        x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                        clf.fit(x_train, y_train)

                        string_maker=string_maker+"""clf = SVC(gamma=n1, C=n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""
                        pred = st.checkbox("Want to see the predictions ?",key=0)
                        prediction = clf.predict(x_test)
                        if pred:
                            st.write(prediction)
                            string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)

                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    string_maker=string_maker+"""reader = pd.read_excel(files)

x=reader.drop(str1,axis=1)
y=reader[str1]\n"""
                    st.dataframe(reader)
                    if n != 0 and n1 != 0:
                        clf = SVC(gamma=n1, C=n)
                        rs_input = st.number_input("Enter the random state", key=1)
                        ts_input = st.number_input("Enter the train test split size ", key=2)
                        x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                        clf.fit(x_train, y_train)
                        string_maker=string_maker+"""clf = SVC(gamma=n1, C=n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""
                        pred = st.checkbox("Want to see the predictions ?",key=0)
                        prediction = clf.predict(x_test)
                        if pred:
                            st.write(prediction)
                            string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)

                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    string_maker=string_maker+"""reader = pd.read_csv(files)

x=reader.drop(str1,axis=1)
y=reader[str1]\n"""
                    st.dataframe(reader)
                    if n != 0 and n1 != 0:
                        clf = SVC(gamma=n1, C=n)
                        rs_input = st.number_input("Enter the random state", key=1)
                        ts_input = st.number_input("Enter the train test split size ", key=2)
                        x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                        clf.fit(x_train, y_train)
                        string_maker=string_maker+"""clf = SVC(gamma=n1, C=n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                        pred = st.checkbox("Want to see the predictions ?",key=0)
                        prediction = clf.predict(x_test)
                        if pred:
                            st.write(prediction)
                            string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ")
                        code_edition = st.checkbox("Download .py edition of the code ")
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")



        elif s == "Gaussian Process":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report,plot_confusion_matrix"""
            n = st.number_input("Enter the Kernel value", key=0)
            n1=st.number_input("Enter the RBF value ",key=3)
            str1=st.text_input("Enter the target column")
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker="""n = int(input("Enter the Kernel value"))
            n1=int(input("Enter the RBF value "))
            str1=input("Enter the target column")
            files = open() #Specify the path of the data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    kernel=n*RBF(n1)
                    clf = GaussianProcessClassifier(kernel=kernel)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
kernel=n*RBF(n1)
clf = GaussianProcessClassifier(kernel=kernel)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    kernel=n*RBF(n1)
                    clf = GaussianProcessClassifier(kernel=kernel)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker=string_maker+"""reader = pd.read_excel(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
kernel=n*RBF(n1)
clf = GaussianProcessClassifier(kernel=kernel)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")




                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    kernel = n * RBF(n1)
                    clf = GaussianProcessClassifier(kernel=kernel)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
kernel=n*RBF(n1)
clf = GaussianProcessClassifier(kernel=kernel))
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


        elif s == "Decision Tree":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,plot_confusion_matrix"""
            n = st.number_input("Enter the Max depth ", key=0)
            # n1 = st.number_input("Enter the RBF value ", key=3)
            str1=st.text_input("Enter the target column")
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker="""n = int(input("Enter the Max depth "))
str1=input("Enter the target column")
files = open() #Specify the path of your data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = DecisionTreeClassifier(max_depth=n)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = DecisionTreeClassifier(max_depth=n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = DecisionTreeClassifier(max_depth=n)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker=string_maker+"""reader = pd.read_excel(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = DecisionTreeClassifier(max_depth=n)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = DecisionTreeClassifier(max_depth=n)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = DecisionTreeClassifier(max_depth=n)
rs_input = int(input("Enter the random state",))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


        elif s == "Random Forest":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix"""
            n = st.number_input("Enter the Max depth ", key=0)
            n1 = st.number_input("Enter the estimators value ", key=3)
            n2 = st.number_input("Enter the Max features value ", key=4)
            str1=st.text_input("Enter the target column")
            string_maker="""n = int(input("Enter the Max depth "))
n1 = int(input("Enter the estimators value "))
n2 = int(input("Enter the Max features value "))
str1=st.text_input("Enter the target column")\n"""

            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker=string_maker+"""files = open() #Specify the path of your data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = RandomForestClassifier(max_depth=n,n_estimators=n1,max_features=n2)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = RandomForestClassifier(max_depth=n,n_estimators=n1,max_features=n2)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""


                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = RandomForestClassifier(max_depth=n,n_estimators=n1,max_features=n2)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_excel(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = RandomForestClassifier(max_depth=n,n_estimators=n1,max_features=n2)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")



                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = RandomForestClassifier(max_depth=n,n_estimators=n1,max_features=n2)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = RandomForestClassifier(max_depth=n,n_estimators=n1,max_features=n2)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

        elif s == "Neural Net":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix"""
            n = st.number_input("Enter the Max depth ", key=0)
            n1 = st.number_input("Enter the RBF value ", key=3)
            str1=st.text_input("Enter the target column")
            string_maker="""n = int(input("Enter the Max depth "))\n
n1 = int(input("Enter the RBF value "))\n
str1=input("Enter the target column")\n"""
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker=string_maker+"""files=open() #Specify the path of your data"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = MLPClassifier(alpha=n,max_iter=n1)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = MLPClassifier(alpha=n,max_iter=n1)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = MLPClassifier(alpha=n,max_iter=n1)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_excel(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = MLPClassifier(alpha=n,max_iter=n1)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(reader, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = MLPClassifier(alpha=n,max_iter=n1)
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = MLPClassifier(alpha=n,max_iter=n1)
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

        elif s == "AdaBoost":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import plot_confusion_matrix"""
            str1 = st.text_input("Enter the target column")
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker="""str1 = st.text_input("Enter the target column")\n
files = open() #Specify the path of your data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = AdaBoostClassifier()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = AdaBoostClassifier()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    reader = pd.get_dummies(reader)
                    x = reader.drop(str1, axis=1)
                    y = reader[str1]
                    st.dataframe(reader)
                    clf = AdaBoostClassifier()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=int(rs_input), test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = AdaBoostClassifier()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_excel(files)
                    st.dataframe(reader)
                    reader = pd.get_dummies(reader)
                    x = reader.drop(str1, axis=1)
                    y = reader[str1]
                    st.dataframe(reader)
                    clf = AdaBoostClassifier()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=int(rs_input), test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_excel(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = AdaBoostClassifier()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

        elif s == "Naive Bayes":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,plot_confusion_matrix"""
            str1 = st.text_input("Enter the target column")
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker = """str1 = st.text_input("Enter the target column")\n
files = open() #Specify the path of your data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = GaussianNB()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianNB()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    reader = pd.get_dummies(reader)
                    x = reader.drop(str1)
                    y = reader[str1]
                    st.dataframe(reader)
                    clf = GaussianNB()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=int(rs_input), test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker = string_maker + """reader = pd.read_excel(files)
x=reader.drop(str1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianNB()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_excel(files)
                    reader = pd.get_dummies(reader)
                    x = reader.drop(str1)
                    y = reader[str1]
                    st.dataframe(reader)
                    clf = GaussianNB()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=int(rs_input), test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker = string_maker + """reader = pd.read_excel(files)
x=reader.drop(str1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianNB()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")
        elif s == "QDA":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report,plot_confusion_matrix"""
            str1 = st.text_input("Enter the target column")
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker="""str1 = input("Enter the target column")\n
            files = open() #Specify the path of your data \n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = QuadraticDiscriminantAnalysis()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = QuadraticDiscriminantAnalysis()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")





                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = QuadraticDiscriminantAnalysis()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker=string_maker+"""reader = pd.read_excel(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = QuadraticDiscriminantAnalysis()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = QuadraticDiscriminantAnalysis()
                    rs_input = st.number_input("Enter the random state", key=1)
                    ts_input = st.number_input("Enter the train test split size ", key=2)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = QuadraticDiscriminantAnalysis()
rs_input = int(input("Enter the random state"))
ts_input = int(input("Enter the train test split size "))
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)
string_maker=string_maker\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ")
                    code_edition = st.checkbox("Download .py edition of the code ")
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")































            # else:
            #     st.write("Kernel size or input size not given ")
            # i will use suggestion system where is needed
















