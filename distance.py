import numpy as np

class Distance:
  def __init__(self):
    pass
  def findCosineDistance(self, source_representation, test_representation):
      a = np.matmul(np.transpose(source_representation), test_representation)
      b = np.sum(np.multiply(source_representation, source_representation))
      c = np.sum(np.multiply(test_representation, test_representation))
      return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

  def findEuclideanDistance(self, source_representation, test_representation):
      if type(source_representation) == list:
          source_representation = np.array(source_representation)

      if type(test_representation) == list:
          test_representation = np.array(test_representation)

      euclidean_distance = source_representation - test_representation
      euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
      euclidean_distance = np.sqrt(euclidean_distance)
      return euclidean_distance

  def l2_normalize(self,x):
      return x / np.sqrt(np.sum(np.multiply(x, x)))
  def findDistance(self,img1_encode, img2_encode, distance_name):
    if distance_name == "cosine":
      dst = self.findCosineDistance(img1_encode, img2_encode)
    elif distance_name == "euclidean":
      dst = self.findEuclideanDistance(img1_encode, img2_encode)
    elif distance_name == "l2_euclidean":
      dst = self.findEuclideanDistance(self.l2_normalize(img1_encode),self.l2_normalize(img2_encode) )
    return dst
