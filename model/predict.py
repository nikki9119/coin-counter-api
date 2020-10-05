import tensorflow as tf
import numpy as np
# model = tf.keras.models.load_model('model/my_model.h5')
with open('model/model_config.json') as json_file:
    json_config = json_file.read()
    model = tf.keras.models.model_from_json(json_config)
model.load_weights('model/rps_weights.h5')
def predict(data):
    x = tf.keras_preprocessing.image.img_to_array(data)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = model.predict(images)
    
    # You may want to further format the prediction to make it more
    # human readable
    return prediction

    # img = image.load_img(img_path, target_size=(300, 300))

    # Preprocessing the image
    
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    # preds = model.predict(images,batch_size=10)
    # return preds