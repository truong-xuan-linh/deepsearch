import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
from distance import Distance

from annoy import AnnoyIndex
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

from tensorflow.keras.applications.vgg19 import VGG19 as vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB4 as eff4
from tensorflow.keras.applications.efficientnet import preprocess_input as eff4_preprocess_input

from tensorflow.keras.models import Model
import tensorflow as tf
os.environ["DEEP-SEARCH"] = "/content/drive/MyDrive/Hahalolo/LOCATION"

class LoadData:
    def __init__(self):
        pass
    def from_folder(self, folder_list:list): # Enter the Single Folder Path/List of the Folders
        self.folder_list = folder_list
        image_path = []
        for folder in self.folder_list:
            for path in os.listdir(folder):
                image_path.append(os.path.join(folder,path))
        return image_path # Returning list of images

class FeatureExtractor:
    def __init__(self, model_name:str):
        self.model_name = model_name
        self.image_size = None
        self.preprocess_input = None
        if self.model_name == "vgg16":
          self.image_size = (224,224)
        # Use VGG-16 as the architecture and ImageNet for the weight
          base_model = vgg16(weights='imagenet')
          self.preprocess_input = vgg16_preprocess_input
          # Customize the model to return features from fully-connected layer
          self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        elif self.model_name == "vgg19":
          self.image_size = (224,224)
          base_model = vgg19(weights='imagenet')
          self.preprocess_input = vgg19_preprocess_input
          # Customize the model to return features from fully-connected layer
          self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        elif self.model_name == "efficientnetb4":
          self.image_size = (380,380)
          base_model = eff4(weights='imagenet')
          self.preprocess_input = eff4_preprocess_input
          # Customize the model to return features from fully-connected layer
          self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
  
    def extract(self, img):
        # Resize the image
        img = img.resize(self.image_size)
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature
    def get_feature(self,image_data:list):
        self.image_data = image_data 
        #fe = FeatureExtractor()
        features = {}
        for img_path in tqdm(self.image_data): # Iterate through images 
            # Extract Features
            img_name = img_path.split("/")[-1]
            #try:
            feature = self.extract(img=Image.open(img_path))
            features[img_name] = feature
            # except:
            #     continue
        return features

class CreateData:
  def __init__(self):
    self.root_dir = os.environ["DEEP-SEARCH"]
  def create(self, image_list, model_name):
    
    self.image_list = image_list
    self.model_name = model_name
    self.FE = FeatureExtractor(self.model_name)
    try:
      os.makedirs(self.root_dir + f"/{self.model_name}-data-files")
      features = self.FE.get_feature(self.image_list)
      a_file = open(self.root_dir + f"/{self.model_name}-data-files/image_encode_data.pkl", "wb")
      pickle.dump(features, a_file)
      a_file.close()

      data = list(features.values())
      t = AnnoyIndex(len(data[0]), 'euclidean')
      for i,v in enumerate(data):
        t.add_item(i, v)
      t.build(100) # 100 trees
      t.save(self.root_dir + f"/{self.model_name}-data-files/image_encode_vector.ann")
    except:
      print("Folder already exist")

class SearchImage:
    def __init__(self):
      self.root_dir = os.environ["DEEP-SEARCH"]
    def search_by_distance(self, model_name, image, k = 10, threshold = 100):
      self.model_name = model_name
      self.image_data = pd.read_pickle(self.root_dir + f"/{self.model_name}-data-files/image_encode_data.pkl")
      rs_images_name = []
      rs_distance = []
    
      images_name = list(self.image_data.keys())
      encodes = list(self.image_data.values())
      fe = FeatureExtractor(self.model_name)
      distance = Distance()
      img_encode = fe.extract(image)
      
      for (i,encode) in enumerate(encodes):
        dst = distance.findDistance(encode, img_encode, "l2_euclidean")
        if dst < threshold:
          rs_images_name.append(images_name[i])
          rs_distance.append(dst)
      rs_distance, rs_images_name = zip(*sorted(zip(rs_distance, rs_images_name)))
      return rs_images_name[:k]
    def search_by_ann(self, model_name, image, k =10):
      
      self.model_name = model_name
      self.k = k # number of output 
      self.image_data = pd.read_pickle(self.root_dir + f"/{self.model_name}-data-files/image_encode_data.pkl")
      self.image_features_vectors_ann = self.root_dir + f"/{self.model_name}-data-files/image_encode_vector.ann"
      self.f = len(list(self.image_data.values())[0])
      fe = FeatureExtractor(self.model_name)
      img_encode = fe.extract(image)
      u = AnnoyIndex(self.f, 'euclidean')
      u.load(self.image_features_vectors_ann) # super fast, will just mmap the file
      index_list = u.get_nns_by_vector(img_encode, self.k) # will find the 10 nearest neighbors
      rs_images_name = []
      names_data = list(self.image_data.keys())
      for idx in index_list:
        rs_images_name.append(names_data[idx])
      return rs_images_name
    def plot_similar_images(self,rs_images_name, root_dir):
        self.root_dir = root_dir
        x = len(rs_images_name)//4 + 1
        # Visualize the result
        axes=[]
        fig=plt.figure(figsize=(20,4*x))
        for a in range(len(rs_images_name)):
            axes.append(fig.add_subplot(x, 4, a+1))  
            plt.axis('off')
            dir = os.path.join(root_dir, "_".join(rs_images_name[a].split("_")[:-1]), rs_images_name[a])
            im = Image.open(dir)
            plt.imshow(im)
        fig.tight_layout()
        plt.show(fig)

