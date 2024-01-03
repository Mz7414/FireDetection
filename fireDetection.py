
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




# 到路徑尋找圖片資料夾並做資料預處理
import tensorflow as tf

normal = 511.0

dataset = tf.keras.utils.image_dataset_from_directory("fire",image_size=(256,256))
dataset = dataset.map(lambda x,y : (x/normal,y*1))
dataset.element_spec

vdataset = tf.keras.utils.image_dataset_from_directory("fire2",image_size=(256,256))
vdataset = vdataset.map(lambda x,y : (x/normal,y*1))
vdataset.element_spec



from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.layers import Dense
import numpy as np

#tf.config.list_physical_devices('GPU') #啟用GPU

m = resnet_v2.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(256,256, 3),
    pooling='max',
    classifier_activation='softmax')

data_bias = np.log(1802./4657)
initializer = tf.keras.initializers.Constant(data_bias)

flattened = Flatten()(m.output)
fc = Dense(4, activation='softmax', bias_initializer=initializer, name="AddedDense2")(flattened)
fc2 = Dropout(0.3)(fc)
model = tf.keras.models.Model(inputs=m.input, outputs=fc2)
model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='drive/MyDrive/best',monitor='val_loss',save_best_only=True)
callback = [checkpoint]

model.fit(dataset, validation_data=vdataset ,batch_size=32, callbacks=callback, epochs=100)

#預測效果
from PIL import Image

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
