#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 下載圖片並儲存
import os

#建立資料夾
path = 'fire'
if not os.path.isdir(path):
    os.mkdir(path)
    os.mkdir(f'{path}/Fire')
    os.mkdir(f'{path}/smoke')
    os.mkdir(f'{path}/parking')

from icrawler.builtin import BingImageCrawler

bing_crawler = BingImageCrawler(
    downloader_threads=4, storage={"root_dir": "fire/Fire"})
bing_crawler.crawl(keyword="fire", max_num=300)

bing_crawler = BingImageCrawler(
    downloader_threads=4, storage={"root_dir": "fire/smoke"}
)
bing_crawler.crawl(keyword="smoke", max_num=300)

bing_crawler = BingImageCrawler(
    downloader_threads=4, storage={"root_dir": "fire/parking"}
)
bing_crawler.crawl(keyword="parking", max_num=300)


# In[ ]:


#建立驗證用資料夾
path = 'fire2'
if not os.path.isdir(path):
    os.mkdir(path)
    os.mkdir(f'{path}/Fire')
    os.mkdir(f'{path}/smoke')
    os.mkdir(f'{path}/parking')

from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    downloader_threads=4, storage={'root_dir': 'fire2/Fire'})
google_crawler.crawl(keyword='fire', max_num=100)

google_crawler = GoogleImageCrawler(
    downloader_threads=4, storage={'root_dir': 'fire2/smoke'})
google_crawler.crawl(keyword='smoke', max_num=100)

google_crawler = GoogleImageCrawler(
    downloader_threads=4, storage={'root_dir': 'fire2/parking'})
google_crawler.crawl(keyword='parking', max_num=100)


# In[ ]:


#檢測資料夾中是否有無法讀入的圖片並自動刪除
from pathlib import Path
import imghdr

data_dir = "fire/smoke"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            os.remove(filepath)
        elif img_type not in img_type_accepted_by_tf:
            os.remove(filepath)


# In[118]:


# 到路徑尋找圖片資料夾並做資料預處理
import tensorflow as tf

normal = 511.0

dataset = tf.keras.utils.image_dataset_from_directory("fire",image_size=(256,256))
dataset = dataset.map(lambda x,y : (x/normal,y*1))
dataset.element_spec

vdataset = tf.keras.utils.image_dataset_from_directory("fire2",image_size=(256,256))
vdataset = vdataset.map(lambda x,y : (x/normal,y*1))
vdataset.element_spec
#list(dataset.as_numpy_iterator())


# In[ ]:


from tensorflow.keras import layers, models

#tf.config.list_physical_devices('GPU') #啟用GPU

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10)) 

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(dataset, epochs=10, validation_data=vdataset)


# In[184]:


#預測效果
from PIL import Image
import tensorflow as tf
import numpy as np

im = Image.open("match.jpg") #圖片路徑
y = tf.image.resize(im, (256, 256))
image_resized = tf.reshape(y, (-1, 256, 256, 3))/normal
prediction = model.predict(image_resized)

# 0表示有火 1表示沒有 2表示有煙
a = np.argmax(prediction[0])
if a == 0:
    print("fire")
elif a == 1:
    print("NoFire")
else:
    print("smoke")

