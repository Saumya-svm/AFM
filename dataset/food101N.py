import torch.utils.data as data
from PIL import Image
import pandas as pd

def read_list(root, fileList):
    imgList = []
    c_dict = []
    k=-1
    imagelist = pd.read_csv('data/food/Food-101N_release/meta/imagelist.tsv', sep='/')
    labels = imagelist['class_name'].unique()
    label_map = {label:i for i, label in enumerate(labels)}
    grouped = imagelist.groupby('class_name')
    subgroup = grouped.apply(lambda x: x.head(100))
    for i,row in enumerate(subgroup.itertuples()):
      imgPath = root+'/images'+str(row.key)+'.jpg'
      label = label_map[row.class_name]
      imgList.append((imgPath,label))
    return imgList

    with open(root + '/' + fileList, 'r') as file:
        for line in file.readlines():
            row = line.strip().split('\t')
            if len(row) == 1:
                label_name, _ = row[0].strip().split('/')
                imgP = line.strip()

                if label_name == 'class_name':
                    continue
                else:
                    if not label_name in c_dict:
                        k += 1
                        c_dict.append(label_name)
                        label = k

                    else:
                        label = k
                    imgPath = root + '/images/' + imgP
                    imgList.append((imgPath, int(label)))
            else:
                imgP = row[0]
                label_name, _ = row[0].strip().split('/')

                if label_name == 'class_name':
                    continue
                else:
                    if not label_name in c_dict:
                        k += 1
                        c_dict.append(label_name)
                        label = k

                    else:
                        label = k
                    imgPath = root + '/images/' + imgP
                    imgList.append((imgPath, int(label)))

    return imgList

class Food101N(data.Dataset):
    def __init__(self, root, transform):
        self.imgList = read_list(root, 'meta/imagelist.tsv')
        self.transform = transform

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = Image.open(imgPath)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)
  
if __name__ == '__main__':
  imgList = read_list('data/food/Food-101N_release', 'imagelist.tsv')
  print(len(imgList))
