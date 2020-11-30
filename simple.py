import tensorflow as tf
import os
from tensorflow.keras import layers

# Dataset information
img_height = 224
img_width = 224
image_size = [img_height, img_width]
class_names = ['cat', 'dog']
num_classes = 2
total_train_image_quantity = 25000

AUTOTUNE = tf.data.experimental.AUTOTUNE

path = 'dataset/train'
test_path = 'dataset/test1'
batch_size = 32

# TODO 파일 경로를 tfds로 가져오세요
data_list = tf.data.Dataset.list_files(os.path.join(path, 'train/*'), shuffle=True)
test_list = tf.data.Dataset.list_files(os.path.join(test_path, 'test1/*'), shuffle=False)


# TODO 파일 경로를 (img, Label)로 바꾸세요
def get_label(file_path):
    parts = tf.strings.split(file_path, '\\')
    one_hot = tf.strings.split(parts[-1], '.')[0] == class_names
    return tf.where(one_hot)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])


def preprocess(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.cast(img, tf.float32)/255.0
    return img, label


# TODO 파일 경로를 (img, label)로 바꾸세요
train_list = data_list.skip(int(total_train_image_quantity / 10))
validation_list = data_list.take(int(total_train_image_quantity / 10))

train_ds = train_list.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = validation_list.map(preprocess, num_parallel_calls=AUTOTUNE)


# TODO Configuration for performance
def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

# 모델 저장
