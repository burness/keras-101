from keras.models import model_from_json
from PIL import Image
import numpy as np
json_file = 'dog-cat.model.json'
model_file = 'dog-cat.model.h5'
image_files = ['11672.jpg', '3213.jpg','3216.jpg','3217.jpg']

json_file = open(json_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_file)
print("Loaded model from disk")


all_images = []
for image_file in image_files:
    im = Image.open(image_file).resize((224, 224))
    im = np.array(im) / 255.0
    all_images.append(im)

all_images = np.array(all_images)
all_images = all_images.reshape(-1, 224, 224, 3)

result = loaded_model.predict(all_images)
print result
# for i, temp in enumerate(loaded_model.layers):
#     print i 
#     print temp


# evaluate loaded model on test data
