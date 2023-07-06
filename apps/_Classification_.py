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
from sklearn.metrics import classification_report
import cv2 as cv
from io import StringIO
import sys



def app():
    st.write("# Classification")

    selector=st.selectbox("Choose Machine Learning Library",("Sklearn","Tensorflow"),key=0)
    if selector=="Tensorflow":
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        selector1=st.selectbox("Choose Classifier",("Multi-class","Binary"),key=1)

        n_inputs=st.number_input("Enter the number of neurons",value=0,key=47)
        n_inputs1 = st.number_input("Enter the number of Dense layers", value=0, key=49)

        n_inputs3=st.number_input("Enter the input size",key=3,value=0)
        slide_input=st.slider("Choose striding limit",key=48)
        slide_input1 = st.slider("Choose color type like 1 for binary and 3 for RGB", key=50,max_value=5,min_value=1)
        string_make="""import numpy as np
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from PIL import  Image
import cv2 as cv
from sklearn.model_selection import train_test_split
from keras.preprocessing import image #Optional...! use this when you use image classification"""
        string_maker = """n_inputs=""" + str(n_inputs)
        string_maker = string_maker + "\n" + "n_inputs1=" + str(n_inputs1)
        string_maker = string_maker + "\n" + "n_inputs3=" + str(n_inputs3)
        string_maker = string_maker + "\n" + "slide_input=" + str(slide_input)
        string_maker = string_maker + "\n" + "slide_input1=" + str(slide_input1)+"\n"



        if n_inputs3 and slide_input:
            model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(n_inputs,(slide_input,slide_input),activation="relu",input_shape=(n_inputs3,n_inputs3,slide_input1))])
            string_maker=string_maker+"""model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(n_inputs,(slide_input,slide_input),activation="relu",input_shape=(n_inputs3,n_inputs3,slide_input1))])\n"""

            check=st.checkbox("Select this to apply dynamic neural networks and unselect for static",key=51)
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
                selector3 = st.selectbox("Choose optimizers", ("SGD", "Adam", "RmsProp"), key=55)
                apply=st.checkbox("Apply custom learning rate",key=56)
            # if n_inputs3 and slide_input:
                if apply:
                    n_inputs2 = st.number_input("Enter the learning rate of neurons", key=52)
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


                rs_input=st.number_input("Enter the random state of the model",value=0,key=59)
                split_input=st.number_input("Enter the amount fo testing samples",value=0,key=58)
                string_maker = string_maker + """rs_input=""" + str(rs_input)
                string_maker = string_maker + "\n" + """"split_input=""" + str(split_input)


                selector4=st.selectbox("Choose classification type",("Image","DataFrame"),key=57)
                if selector4=="Image":
                    files=st.file_uploader("Upload first item",type=["png","jpg","jpeg"],key=207,accept_multiple_files=True)
                    name=st.text_input("Name of the first item",key=208)
                    # files2=st.file_uploader()
                    train_data = []
                    train_data1=[]
                    files1=st.file_uploader("Upload second item",type=["png","jpg","jpeg"],key=60,accept_multiple_files=True)
                    name1=st.text_input("Name of the second item",key=61)
                    st.write("Enter the prediction data")
                    files2 = st.file_uploader("Upload files", type=["jpg", "png", "jpeg"], key=62,accept_multiple_files=True)
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

                    
                    for i in files:
                        img=Image.open(i)
                        img_array=np.array(img)
                        img_new=cv.resize(img_array,(n_inputs3,n_inputs3))
                        train_data.append([img_new,0])
                    for j in files1:
                        img1=Image.open(j)
                        img1_array=np.array(img1)
                        img_new1=cv.resize(img1_array,(n_inputs3,n_inputs3))
                        train_data.append([img_new1,1])
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


                    
                    st.write(features.shape)
                    X=np.array(X).reshape(-1,n_inputs3,n_inputs3,slide_input1)
                    Y=np.array(Y)

                    x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=int(rs_input),test_size=split_input/100)


                    batch=st.checkbox("Want to keep batch size same as random state...",key=209)

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
                    text_edition = st.checkbox("Download .txt edition of the code ", key=210)
                    code_edition = st.checkbox("Download .py edition of the code ", key=211)
                    string_total = string_make + """\n""" + string_maker
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                    button22 = st.checkbox("Start predicting", key=212)
                    if button22:
                        if batch:
                            epoch=st.number_input("Enter the number of epochs ",value=0,key=213)
                            validation=st.number_input("Enter the number of validation steps",value=0,key=214)
                            step=st.number_input("Enter steps per epochs",value=0,key=215)

                            batch_num1=rs_input
                            if epoch !=0:
                                model.fit(x_train,y_train,batch_size=batch_num1,epochs=epoch,steps_per_epoch=int(step),validation_data=(x_test,y_test),validation_steps=validation)
                                classes = model.predict(images, batch_size=batch_num1)
                                if classes[0] > 0:
                                    st.write("is a", name)
                                    model.summary()
                                    sys.stdout = old_stdout
                                    st.text(mystdout.getvalue())

                                else:
                                    st.write("is a", name1)
                                    model.summary()
                                    sys.stdout = old_stdout
                                    st.text(mystdout.getvalue())


                        else:
                            batch_num=st.number_input("Enter the amount of batch size ",value=1,key=216)
                            epoch = st.number_input("Enter the number of epochs ", value=0, key=217)
                            validation = st.number_input("Enter the number of validation steps", value=0, key=218)
                            step = st.number_input("Enter steps per epochs", value=0, key=219)
                            batch_num1=batch_num
                            if epoch !=0 :
                                model.fit(x_train,y_train,batch_size=batch_num1,epochs=epoch,steps_per_epoch=int(step),validation_data=(x_test,y_test),validation_steps=validation)
                                classes = model.predict(images, batch_size=batch_num1)
                                if classes[0] > 0:
                                    st.write("is a", name)
                                    model.summary()
                                    sys.stdout = old_stdout
                                    st.text(mystdout.getvalue())

                                else:
                                    st.write("is a", name1)
                                    model.summary()
                                    sys.stdout = old_stdout
                                    st.text(mystdout.getvalue())


                            


                    elif selector4=="DataFrame":
                        files = st.file_uploader("Upload first item", type=["csv", "xls", "xlsx"], key=220,accept_multiple_files=True)
                        str1 = st.text_input("Enter the target column", key=221)
                        string_maker=string_maker+"""files=() # Specify the path for your data \n"""
                        string_maker=string_maker+"""str1 = input("Enter the target column")"""
                        if files is not None:
                            if files.name.split(".")[1]=="csv":
                                data=pd.read_csv(files)
                                data=pd.get_dummies(data)
                                x=data.drop(str1,axis=1)
                                y=data[str1]
                                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=split_input / 100)
                                epoch = st.number_input("Enter the number of epochs ", value=1, key=222)
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
                                model.summary()
                                sys.stdout = old_stdout
                                st.text(mystdout.getvalue())
                                string_maker=string_maker+"""prediction = model.predict(x_test)"""


                                string_total = string_make + """\n""" + string_maker
                                text_edition = st.checkbox("Download .txt edition of the code ", key=223)
                                code_edition = st.checkbox("Download .py edition of the code ", key=224)
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
                                epoch = st.number_input("Enter the number of epochs ", value=1, key=225)
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
                                model.summary()
                                sys.stdout = old_stdout
                                st.text(mystdout.getvalue())
                                string_maker=string_maker+"""prediction = model.predict(x_test)\n"""


                                string_total = string_make + """\n""" + string_maker
                                text_edition = st.checkbox("Download .txt edition of the code ", key=226)
                                code_edition = st.checkbox("Download .py edition of the code ", key=227)
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

                                epoch = st.number_input("Enter the number of epochs ", value=1, key=228)
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
                                model.summary()
                                sys.stdout = old_stdout
                                st.text(mystdout.getvalue())
                                string_maker=string_maker+"""prediction=model.predict(x_test)\n"""

                                string_total = string_make + """\n""" + string_maker
                                text_edition = st.checkbox("Download .txt edition of the code ", key=229)
                                code_edition = st.checkbox("Download .py edition of the code ", key=230)
                                if text_edition:
                                    st.download_button(label="Download", data=string_total, file_name="text/.txt")

                                elif code_edition:
                                    st.download_button(label="Download", data=string_total, file_name="python/.py")




            elif selector1=="Multi-class":
                model.add(tf.keras.layers.Dense(n_inputs * n_inputs1, activation="relu"))
                model.add(tf.keras.layers.Flatten())
                values=st.number_input("Enter the number of classes",key=240)
                model.add(tf.keras.layers.Dense(values, activation="softmax"))

                string_maker=string_maker+"""model.add(tf.keras.layers.Dense(n_inputs * n_inputs1, activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation="softmax"))\n"""

                selector2 = st.selectbox("Choose loss function", (
                "binary_cross_entropy", "categorical_cross_entropy", "sparse_categorical_cross_entropy"))
                selector3 = st.selectbox("Choose optimizers", ("SGD", "Adam", "RmsProp"), key=53)
                apply = st.checkbox("Apply custom learning rate", key=242)
                # if n_inputs3 and slide_input:
                if apply:
                    n_inputs2 = st.number_input("Enter the learning rate of neurons", key=241)
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


                rs_input = st.number_input("Enter the random state of the model", value=0, key=243)
                split_input = st.number_input("Enter the amount fo testing samples", value=0, key=244)
                string_maker = string_maker + """rs_input=""" + str(rs_input)
                string_maker = string_maker + "\n" + """"split_input=""" + str(split_input)

                train_data = []

                name_appender = []
                string_maker=string_maker+""" train_data = []\n
values=int(input("Enter the number of classes"))\n
name_appender = []\n"""
                val_count=0
                for val in range(700):
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
                files2 = st.file_uploader("Upload files", type=["jpg", "png", "jpeg"], key=54,accept_multiple_files=True)
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
                    text_edition = st.checkbox("Download .txt edition of the code ", key=22)
                    code_edition = st.checkbox("Download .py edition of the code ", key=21)
                    string_total = string_make + """\n""" + string_maker
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                    button22=st.checkbox("Start predicting",key=20)
                    if button22:
                        batch = st.checkbox("Want to keep batch size same as random state...",key=0)
                        if batch:
                            epoch = st.number_input("Enter the number of epochs ", value=0, key=6)
                            validation = st.number_input("Enter the number of validation steps",key=21)

                            batch_num1 = rs_input
                            if epoch != 0 :
                                model.fit(x_train, y_train, batch_size=batch_num1, validation_data=(x_test, y_test), epochs=epoch,validation_steps=validation)
                                classes = model.predict(images, batch_size=batch_num1)
                                while ab < len(name_appender):
                                    if classes[0] > name_appender.index(name_appender[0]):
                                        st.write("is a", name_appender[ab])
                                        model.summary()
                                        sys.stdout = old_stdout
                                        st.text(mystdout.getvalue())
                                        break
                                    else:
                                        st.write("is a", name_appender[0])
                                        model.summary()
                                        sys.stdout = old_stdout
                                        st.text(mystdout.getvalue())
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
                                        st.write("is a", name_appender[ab])
                                        model.summary()
                                        sys.stdout = old_stdout
                                        st.text(mystdout.getvalue())
                                        break
                                    else:
                                        st.write("is a", name_appender[0])
                                        model.summary()
                                        sys.stdout = old_stdout
                                        st.text(mystdout.getvalue())
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
    "QDA",),key=45)
        if s=="Nearest Neighbors":
            string_make="""import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix"""
            n=st.number_input("Enter the number neighbours",key=46)

            files=st.file_uploader("Choose files",type=["xls","xlsx","csv"])
            string_maker = """n="""+str(n)+"\n"
            string_maker=string_maker+"\n"+"""
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
                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = KNeighborsClassifier(int(n))\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    prediction = clf.predict(x_test)
                    pred = st.checkbox("Want to see the predictions ?", key=205)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test,prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=22)
                    code_edition = st.checkbox("Download .py edition of the code ", key=21)
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

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf=KNeighborsClassifier(int(n))\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test,prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=22)
                    code_edition = st.checkbox("Download .py edition of the code ", key=21)
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
                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = KNeighborsClassifier(int(n))\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=0)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=22)
                    code_edition = st.checkbox("Download .py edition of the code ", key=21)
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
            n = st.number_input("Enter the Regularization value", key=63)

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
                    rs_input = st.number_input("Enter the random state", key=64)
                    ts_input = st.number_input("Enter the train test split size ", key=65)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),
                                                                        test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker=string_maker+"""reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(kernel="linear",C=n)\n"""
                    string_maker=string_maker+"rs_input="+str(rs_input)+"\n"
                    string_maker=string_maker+"ts_input="+str(ts_input)+"\n"
                    string_maker=string_maker+"""x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=66)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=22)
                    code_edition = st.checkbox("Download .py edition of the code ", key=21)
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
                    rs_input = st.number_input("Enter the random state", key=67)
                    ts_input = st.number_input("Enter the train test split size ", key=68)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),
                                                                        test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(kernel="linear",C=n)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=69)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=77)
                    code_edition = st.checkbox("Download .py edition of the code ", key=78)
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
                    rs_input = st.number_input("Enter the random state", key=70)
                    ts_input = st.number_input("Enter the train test split size ", key=71)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(kernel="linear",C=n)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=76)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=74)
                    code_edition = st.checkbox("Download .py edition of the code ", key=75)
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
            n = st.number_input("Enter the Regularization value", key=72)
            n1=st.number_input("Enter the gamma value ",key=73)
            string_maker="""n = int(input("Enter the Regularization value"))\n
n1=int(input("Enter the gamma value "))\n"""
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"],key=206)
            str1=st.text_input("Enter the target column")
            string_maker=string_maker+"""files = open() #Specify the path of the data\n
str1=input("Enter the target column")\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)

                    if n != 0 and n1 != 0:
                        clf = SVC(gamma=n1, C=n)
                        rs_input = st.number_input("Enter the random state", key=79)
                        ts_input = st.number_input("Enter the train test split size ", key=80)
                        x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                        clf.fit(x_train, y_train)

                        string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(gamma=n1, C=n)\n"""
                        string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                        string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                        string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""


                        pred = st.checkbox("Want to see the predictions ?",key=81)
                        prediction = clf.predict(x_test)
                        if pred:
                            st.write(prediction)
                            st.write(classification_report(y_test, prediction))
                            string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ", key=82)
                        code_edition = st.checkbox("Download .py edition of the code ", key=83)
                        if text_edition:
                            st.download_button(label="Download", data=string_total, file_name="text/.txt")

                        elif code_edition:
                            st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)

                    x=reader.drop(str1,axis=1)
                    y=reader[str1]

                    st.dataframe(reader)
                    if n != 0 and n1 != 0:
                        clf = SVC(gamma=n1, C=n)
                        rs_input = st.number_input("Enter the random state", key=84)
                        ts_input = st.number_input("Enter the train test split size ", key=85)
                        x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                        clf.fit(x_train, y_train)

                        string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(gamma=n1, C=n)\n"""
                        string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                        string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                        string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""


                        pred = st.checkbox("Want to see the predictions ?",key=86)
                        prediction = clf.predict(x_test)
                        if pred:
                            st.write(prediction)
                            st.write(classification_report(y_test, prediction))
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
                    if n != 0 and n1 != 0:
                        clf = SVC(gamma=n1, C=n)
                        rs_input = st.number_input("Enter the random state", key=87)
                        ts_input = st.number_input("Enter the train test split size ", key=88)
                        x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                        clf.fit(x_train, y_train)

                        string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = SVC(gamma=n1, C=n)\n"""
                        string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                        string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                        string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                        pred = st.checkbox("Want to see the predictions ?",key=89)
                        prediction = clf.predict(x_test)
                        if pred:
                            st.write(prediction)
                            string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                        string_total = string_make + """\n""" + string_maker
                        text_edition = st.checkbox("Download .txt edition of the code ", key=90)
                        code_edition = st.checkbox("Download .py edition of the code ", key=91)
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
            n = st.number_input("Enter the Kernel value", key=92)
            n1=st.number_input("Enter the RBF value ",key=93)
            str1=st.text_input("Enter the target column")
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker="n="+str(n)+"\n"
            string_maker=string_maker+"n1="+str(n1)+"\n"
            string_maker=string_maker+"str1="+str(str1)+"\n"
            string_maker=string_maker+"""files = open() #Specify the path of the data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    kernel=n*RBF(n1)
                    clf = GaussianProcessClassifier(kernel=kernel)
                    rs_input = st.number_input("Enter the random state", key=94)
                    ts_input = st.number_input("Enter the train test split size ", key=95)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianProcessClassifier(kernel=kernel)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=96)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=97)
                    code_edition = st.checkbox("Download .py edition of the code ", key=98)
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
                    rs_input = st.number_input("Enter the random state", key=99)
                    ts_input = st.number_input("Enter the train test split size ", key=100)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)
                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianProcessClassifier(kernel=kernel)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=101)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=102)
                    code_edition = st.checkbox("Download .py edition of the code ", key=103)
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
                    rs_input = st.number_input("Enter the random state", key=104)
                    ts_input = st.number_input("Enter the train test split size ", key=105)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianProcessClassifier(kernel=kernel)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=106)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=107)
                    code_edition = st.checkbox("Download .py edition of the code ", key=108)
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
            n = st.number_input("Enter the Max depth ", key=109)
            # n1 = st.number_input("Enter the RBF value ", key=3)
            str1=st.text_input("Enter the target column")
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker="""n ="""+ str(n)+"\n"
            string_maker=string_maker+"""str1="""+str(str1)+"\n"
            string_maker=string_maker+"""files = open() #Specify the path of your data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = DecisionTreeClassifier(max_depth=int(n))
                    rs_input = st.number_input("Enter the random state", key=110)
                    ts_input = st.number_input("Enter the train test split size ", key=111)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = DecisionTreeClassifier(max_depth=n)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=112)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=113)
                    code_edition = st.checkbox("Download .py edition of the code ", key=114)
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = DecisionTreeClassifier(max_depth=int(n))
                    rs_input = st.number_input("Enter the random state", key=115)
                    ts_input = st.number_input("Enter the train test split size ", key=116)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = DecisionTreeClassifier(max_depth=n)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=117)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=118)
                    code_edition = st.checkbox("Download .py edition of the code ", key=119)
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = DecisionTreeClassifier(max_depth=int(n))
                    rs_input = st.number_input("Enter the random state", key=120)
                    ts_input = st.number_input("Enter the train test split size ", key=121)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = DecisionTreeClassifier(max_depth=n)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=122)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=123)
                    code_edition = st.checkbox("Download .py edition of the code ", key=124)
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
            n = st.number_input("Enter the Max depth ", key=125)
            n1 = st.number_input("Enter the estimators value ", key=126)
            n2 = st.number_input("Enter the Max features value ", key=127)
            str1=st.text_input("Enter the target column")
            string_maker="""n ="""+ str(n)+"\n"
            string_maker=string_maker+"n1 ="+str(n1)+"\n"
            string_maker=string_maker+"n2 ="+ str(n2)+"\n"
            string_maker=string_maker+"str1="+str(str1)+"\n"

            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker=string_maker+"""files = open() #Specify the path of your data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = RandomForestClassifier(max_depth=int(n),n_estimators=int(n1),max_features=int(n2))
                    rs_input = st.number_input("Enter the random state", key=128)
                    ts_input = st.number_input("Enter the train test split size ", key=129)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = RandomForestClassifier(max_depth=n,n_estimators=n1,max_features=n2)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""


                    pred = st.checkbox("Want to see the predictions ?", key=130)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=131)
                    code_edition = st.checkbox("Download .py edition of the code ", key=132)
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = RandomForestClassifier(max_depth=int(n),n_estimators=int(n1),max_features=int(n2))
                    rs_input = st.number_input("Enter the random state", key=133)
                    ts_input = st.number_input("Enter the train test split size ", key=134)
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

                    pred = st.checkbox("Want to see the predictions ?", key=135)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=136)
                    code_edition = st.checkbox("Download .py edition of the code ", key=137)
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")



                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = RandomForestClassifier(max_depth=int(n),n_estimators=int(n1),max_features=int(n2))
                    rs_input = st.number_input("Enter the random state", key=138)
                    ts_input = st.number_input("Enter the train test split size ", key=139)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = RandomForestClassifier(max_depth=n,n_estimators=n1,max_features=n2)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=140)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=141)
                    code_edition = st.checkbox("Download .py edition of the code ", key=142)
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
            n = st.number_input("Enter the Max depth ", key=143)
            n1 = st.number_input("Enter the RBF value ", key=144)
            str1=st.text_input("Enter the target column")
            string_maker="""n ="""+ str(n)+"\n"
            string_maker=string_maker+"n1 ="+ str(n1)+"\n"
            string_maker=string_maker+"str1="+str(str1)
            files = st.file_uploader("Choose files", type=["xls", "xlsx", "csv"])
            string_maker=string_maker+"""files=open() #Specify the path of your data"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = MLPClassifier(alpha=int(n),max_iter=int(n1))
                    rs_input = st.number_input("Enter the random state", key=145)
                    ts_input = st.number_input("Enter the train test split size ", key=146)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = MLPClassifier(alpha=n,max_iter=n1)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=147)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=148)
                    code_edition = st.checkbox("Download .py edition of the code ", key=149)
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")


                elif files.name.split(".")[1] == "xls":
                    reader = pd.read_excel(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = MLPClassifier(alpha=int(n),max_iter=int(n1))
                    rs_input = st.number_input("Enter the random state", key=150)
                    ts_input = st.number_input("Enter the train test split size ", key=151)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = MLPClassifier(alpha=n,max_iter=n1)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=152)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=153)
                    code_edition = st.checkbox("Download .py edition of the code ", key=154)
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")

                elif files.name.split(".")[1] == "xlsx":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = MLPClassifier(alpha=int(n),max_iter=int(n1))
                    rs_input = st.number_input("Enter the random state", key=155)
                    ts_input = st.number_input("Enter the train test split size ", key=156)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = MLPClassifier(alpha=n,max_iter=n1)\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=157)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=158)
                    code_edition = st.checkbox("Download .py edition of the code ", key=159)
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
            string_maker="""str1 ="""+ str(str1)+"\n"
            string_maker=string_maker+"""files = open() #Specify the path of your data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = AdaBoostClassifier()
                    rs_input = st.number_input("Enter the random state", key=160)
                    ts_input = st.number_input("Enter the train test split size ", key=161)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = AdaBoostClassifier()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=162)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=163)
                    code_edition = st.checkbox("Download .py edition of the code ", key=164)
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
                    rs_input = st.number_input("Enter the random state", key=165)
                    ts_input = st.number_input("Enter the train test split size ", key=166)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=int(rs_input), test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = AdaBoostClassifier()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=167)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=168)
                    code_edition = st.checkbox("Download .py edition of the code ", key=169)
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
                    rs_input = st.number_input("Enter the random state", key=170)
                    ts_input = st.number_input("Enter the train test split size ", key=171)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=int(rs_input), test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = AdaBoostClassifier()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=172)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=173)
                    code_edition = st.checkbox("Download .py edition of the code ", key=174)
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
            string_maker="""str1 ="""+ str(str1)+"\n"
            string_maker=string_maker+"""files = open() #Specify the path of your data\n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = GaussianNB()
                    rs_input = st.number_input("Enter the random state", key=175)
                    ts_input = st.number_input("Enter the train test split size ", key=176)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianNB()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=177)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=178)
                    code_edition = st.checkbox("Download .py edition of the code ", key=179)
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
                    rs_input = st.number_input("Enter the random state", key=180)
                    ts_input = st.number_input("Enter the train test split size ", key=181)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=int(rs_input), test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianNB()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=182)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=183)
                    code_edition = st.checkbox("Download .py edition of the code ", key=184)
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
                    rs_input = st.number_input("Enter the random state", key=185)
                    ts_input = st.number_input("Enter the train test split size ", key=186)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=int(rs_input), test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = GaussianNB()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=187)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker = string_maker + """prediction = clf.predict(x_test)\n"""




                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=188)
                    code_edition = st.checkbox("Download .py edition of the code ", key=189)
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
            string_maker="""str1 ="""+ str(str1)+"\n"
            string_maker=string_maker+"""files = open() #Specify the path of your data \n"""
            if files is not None:
                if files.name.split(".")[1] == "csv":
                    reader = pd.read_csv(files)
                    
                    x=reader.drop(str1,axis=1)
                    y=reader[str1]
                    st.dataframe(reader)
                    clf = QuadraticDiscriminantAnalysis()
                    rs_input = st.number_input("Enter the random state", key=190)
                    ts_input = st.number_input("Enter the train test split size ", key=191)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = QuadraticDiscriminantAnalysis()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=192)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=193)
                    code_edition = st.checkbox("Download .py edition of the code ", key=194)
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
                    rs_input = st.number_input("Enter the random state", key=195)
                    ts_input = st.number_input("Enter the train test split size ", key=196)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = QuadraticDiscriminantAnalysis()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=197)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=198)
                    code_edition = st.checkbox("Download .py edition of the code ", key=199)
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
                    rs_input = st.number_input("Enter the random state", key=200)
                    ts_input = st.number_input("Enter the train test split size ", key=201)
                    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
                    clf.fit(x_train, y_train)

                    string_maker = string_maker + """reader = pd.read_csv(files)
x=reader.drop(str1,axis=1)
y=reader[str1]
st.dataframe(reader)
clf = QuadraticDiscriminantAnalysis()\n"""
                    string_maker = string_maker + "rs_input=" + str(rs_input) + "\n"
                    string_maker = string_maker + "ts_input=" + str(ts_input) + "\n"
                    string_maker = string_maker + """x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=int(rs_input),test_size=ts_input/100)
clf.fit(x_train, y_train)\n"""

                    pred = st.checkbox("Want to see the predictions ?", key=202)
                    prediction = clf.predict(x_test)
                    if pred:
                        st.write(prediction)
                        st.write(classification_report(y_test, prediction))
                        string_maker=string_maker+"""prediction = clf.predict(x_test)\n"""



                    string_total = string_make + """\n""" + string_maker
                    text_edition = st.checkbox("Download .txt edition of the code ", key=203)
                    code_edition = st.checkbox("Download .py edition of the code ", key=204)
                    if text_edition:
                        st.download_button(label="Download", data=string_total, file_name="text/.txt")

                    elif code_edition:
                        st.download_button(label="Download", data=string_total, file_name="python/.py")































            # else:
            #     st.write("Kernel size or input size not given ")
            # i will use suggestion system where is needed
















