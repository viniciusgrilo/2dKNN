import math
import numpy as np
import matplotlib.pyplot as plt
import random

class KNN:

    def __init__(self, k:int = 3):
        self._k = k
        self._trained_objects = None
    
    def euclidian_distance(self, p: tuple, q:tuple): #p and q are coordinates: (p0, p1), (q0, q1)
        return float(math.sqrt(
                math.pow(
                    (q[0] - p[0]), 2
                )
                +
                math.pow(
                    (q[1] - p[1]), 2
                )              
            )
        )

    def convert_list_to_nparray(self, list:list):
        return np.array(list)

    def convert_nominal_features_to_nparray(self, features:list):
        #converts classes to integer numpy arrays
        _, _, label_list_idx = np.unique(features, return_index=True, return_inverse=True) 
        return label_list_idx

    def normalize_features(self, features:list):
        #Feature normalization = value - value.mean / value.standard_deviation
        return (features - features.mean(0)) / features.std(0)   

    def train_test_split(self, dataset):
        random.shuffle(dataset)
        #70/30 split
        S = int(len(dataset)*0.7)
        train_dataset = dataset[:S]
        test_dataset = dataset[S+1:]
        return train_dataset, test_dataset        
        
    def find_neighbors(self, v:tuple): 
        neighbors = []
        for obj in self._trained_objects: #calculate de distance between all trained objects
            N = {}
            N["x_axis"] = v[0]
            N["y_axis"] = v[1]
            N["neighbor_x_axis"] = obj["x_axis"]
            N["neighbor_y_axis"] = obj["y_axis"]
            N["distance"] = self.euclidian_distance(v, (obj["x_axis"], obj["y_axis"]))  #add the distance
            N["neighbor_class"] = obj["class"]
            neighbors.append(N) #group all neighbors
        return neighbors

    def train(self, dataset):
        try:
            objects = []
            index = 0
            for item in dataset:
                trained = {}
                trained["index"] = index
                trained["x_axis"] = item[0] #sepal_area
                trained["y_axis"] = item[1] #petal_area
                trained["class"] = item[2] #class
                objects.append(trained)
                index += 1
            self._trained_objects = objects
        except:
            raise SystemExit("Model training error")          

    def predict(self, v:tuple()): #v is the coordinate of the object we want to classify: (v0, v1)   
        if self._trained_objects == None:
            return "Train the model before making a prediction!"
        neighbors = self.find_neighbors(v) #dict
        neighbors.sort(key=lambda k: k["distance"]) #sort the array 
        nearest_neighbors = neighbors[:self._k] #extract the K nearest neighbors 
        return nearest_neighbors

    def plot_data(self, x, y, title, color):
        plt.xlabel("Sepal Area")
        plt.ylabel("Petal Area")
        plt.title(title)
        plt.scatter(x, y, c = color)
        return plt.plot()



