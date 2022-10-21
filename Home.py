import streamlit as st
from KNN import KNN
import pandas as pd

df = pd.read_csv("IRIS.csv")

sepal_length = df["sepal_length"]
sepal_width = df["sepal_width"]
petal_length = df["petal_length"]
petal_width = df["petal_width"]

#calculate area of petals and sepals (so we can word with 2D features intead of 4D)
sepal_area = list(sepal_length * sepal_width)
petal_area = list(petal_length * petal_width)

#instantiate the classifier 
#KNN(k) | k is the number os neighbors
clf = KNN(k = 3)

#convert to numpy arrays
sepal_area = clf.convert_list_to_nparray(sepal_area)
petal_area = clf.convert_list_to_nparray(petal_area)

#normalize the features
sepal_area_norm = clf.normalize_features(sepal_area)
petal_area_norm = clf.normalize_features(petal_area)

#convert labels to numpy arrays
classes = clf.convert_nominal_features_to_nparray(list(df["species"]))

#create the data matrix
# the feature matrix is a 2D array which every line represent one object,
# ie in every line we have all information about one single object.

# e.g. feature_matrix = [
#     [sepal area of object 0, petal area of object 0, class of object 0],
#     [sepal area of object 1, petal area of object 1, class of object 1],
#     [sepal area of object 2, petal area of object 2, class of object 2],
#     [sepal area of object 3, petal area of object 3, class of object 3],
#     ...
#     [sepal area of object n, petal area of object n, class of object n],
# ]

feature_matrix = []
for i in range(0,len(df)):
    sa = sepal_area_norm[i]
    pa = petal_area_norm[i]
    c = classes[i]    
    feature_list = [sa, pa, c]
    feature_matrix.append(feature_list)

#dataset split
train, test = clf.train_test_split(feature_matrix)

#train the model
clf.train(train)

#streamlit front-end
st.title("2dKNN") 

st.write(
    '''
A very simple implementation of the KNN algorithm using the Euclidean Distance to calculate the nearest neighbor. Used to classify values based on 2D features.
    '''
)

st.header("Train dataset")
df_train = pd.DataFrame(train, columns=["sepal_area", "petal_area", "class"])
st.write(df_train)

st.subheader("Classes (labels)")
st.write("0 - Iris-setosa")
st.write("1 - Iris-versicolor")
st.write("2 - Iris-virginica")

st.header("Features")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(clf.plot_data(sepal_area, petal_area, "Original Features", ["green"]))
st.pyplot(clf.plot_data(sepal_area_norm, petal_area_norm, "Normalized Features", ["blue"]))

st.header("Test dataset")
df_test = pd.DataFrame(test, columns=["sepal_area", "petal_area", "class"])
st.write(df_test)

index = 0
for flower in test:   
    predict_test = clf.predict((flower[0], flower[1]))
    df_predicted_test = pd.DataFrame(predict_test)
    with st.container():
        st.write(f"Flower {index}'s expected class is {flower[2]}. Its {clf._k} nearest neighbors are:")
        st.write(df_predicted_test)    
    index += 1







