from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16 
from keras.applications.vgg19 import VGG19

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

#help(InceptionResNetV2)
model = ResNet50(weights='imagenet')
#model = VGG19(weights='imagenet')

img_path = '/home/zhouyang/xxxx.jpg'
img = image.load_img(img_path, target_size=(224, 224))
#img = image.load_img(img_path, target_size=(299, 299))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

