import torch.utils.data as data
from PIL import Image
import pandas as pd

def read_list(root, fileList):
    imgList = []
    c_dict = []
    k=-1
    text_list = pd.read_csv(root + '/'+ fileList, sep='/', header=None, names=['class_name', 'image_loc'])
    labels = text_list['class_name'].unique()
    label_map = {label:i for i, label in enumerate(labels)}
    grouped = text_list.groupby('class_name')
    subgroup = grouped.apply(lambda x: x.head(30))
    for i,row in enumerate(subgroup.itertuples()):
      imgPath = root+'/images'+'/'+row.class_name+'/'+str(row.image_loc)+'.jpg'
      label = label_map[row.class_name]
      imgList.append((imgPath,label))

    return imgList

    with open(root + '/'+ fileList, 'r') as file:
        for line in file.readlines():

          label_name, _ = line.strip().split('/')
          imgP = line.strip()

          if not label_name in c_dict:
              k += 1
              c_dict.append(label_name)
              label = k

          else:
              label = k
          imgPath = root + '/images/' + imgP + '.jpg'
          imgList.append((imgPath, int(label)))

    return imgList

class Food101(data.Dataset):
    def __init__(self, root, transform):
        self.imgList = read_list(root, 'meta/test.txt')
        self.transform = transform

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = Image.open(imgPath)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

if __name__ == '__main__':
  imgList = read_list('data/food/food-101', 'meta/test.txt')
  print(len(imgList))
