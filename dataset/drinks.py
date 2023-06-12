import torch.utils.data as data
from PIL import Image
import os
import json

categories = ['Carbonated_Beverages',
 'Ice_Cream',
 'Energy_Drinks',
 'Water',
 'Dairy_Deli',
 'Milk_Creamer',
 'Tea',
 'Juice',
 'Coffee',
 'Sports_Drinks',
 'Pizza',
 'Frozen_Foods',
 'Beer_Ale_Alcoholic_Cider']

with open('label_map.json', 'r') as file:
  label_map = json.load(file)

def create_list(startIndex, endIndex):
  max = 50
  imgList = []
  count = 0
  for category in categories:
    classes = os.listdir(f'./{category}')
    for label in classes:
      if not label.startswith('.'):
        images = os.listdir(f'./{category}/{label}')

        for image in images[startIndex:endIndex]:
          imgList.append((f'./{category}/{label}/{image}',label_map[label]))
    return imgList

class Drinks(data.Dataset):
  def __init__(self, transform, startIndex=0, endIndex=75):
    self.imgList = create_list(startIndex, endIndex)
    self.transform = transform

  def __getitem__(self, index):
    imgPath, target = self.imgList[index]
    img = Image.open(imgPath)
    img = self.transform(img)
    return img, target

  def __len__(self):
    return len(self.imgList)
  
if __name__ == '__main__':
	imgList = create_list(0,50)
	print(imgList[0][0], imgList[0][1])
