# DO NOT RENAME THIS FILE
# This file enables automated judging
# This file should stay named as `submission.py`

# Import Python Libraries
import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import Model

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from keras.layers import TimeDistributed as td
from keras.layers import Conv2D, Flatten, Dense, ZeroPadding2D, Activation
from keras.layers import MaxPooling2D, Dropout, BatchNormalization, Reshape

# Import helper functions from utils.py
from utils import *

# model1 = keras.models.load_model('model1')

class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """
        # self.model = load_model('example_model.h5')
        self.model = keras.models.load_model('94Model.h5')

    def make_prediction(self, img_path):
        """
        DO NOT RENAME THIS FUNCTION
        This function enables automated judging
        This function should stay named as `make_prediction(self, img_path)`

        INPUT:
            img_path: 
                A string representing the path to an RGB image with dimensions 128x128
                example: `example_images/1.png`
        
        OUTPUT:
            A 4-character string representing how to re-arrange the input image to solve the puzzle
            example: `3120`
        """

        # x_train, y_train = getData("train_data_lc") # 15, 15, 610
        # x_test, y_test = getData("test_data_lc") # 15, 15, 15

        # adam = tf.keras.optimizers.Adam(lr=.001)
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

        x = []
        y = []
        # print(img_path)
        image = Image.open(img_path)
        image = np.array(image).astype('float16')
        image = image / 255 - 0.5

        x.append(get_uniform_rectangular_split(image, 2, 2))

        # x.append(image)
        # y.append([int(i) for i in folderLabel])

        # print(2)

        inp = np.expand_dims(x[0], axis=0)
        out = self.model.predict(inp)[0]
        answer = []
        if (len(set(out.argmax(axis=0))) != 4):
            # create array of maximum predictions
            df = np.array(out)
            maxArray = np.array([[0 for x in range(4)] for y in range(4)])
            for i in range(4):
                max = df.argmax()
                maxRow = max // 4
                maxColumn = max % 4
                for index, row in enumerate(df):
                    if (index != maxRow):
                        df[index][maxColumn] = 0
                df[maxRow] = [0 for x in range(len(row))]

                maxArray[maxRow][maxColumn] = 1

            maxArray = maxArray.transpose()
            for i in range(4):
                answer.append(np.where(maxArray[i,] == 1)[0][0])

        else:
            answer = out.argmax(axis=0).tolist()


        string = ""
        for i in answer:
            string = string + str(i)
        prediction = string

        return prediction
        # prediction = self.model.predict(img_tensor, verbose=False)


# Example main function for testing/development
# Run this file using `python3 submission.py`
if __name__ == '__main__':

    for img_name in glob('test_files/*'):
        # Open an example image using the PIL library
        # example_image = Image.open(img_name)

        # print(img_name)

        # Use instance of the Predictor class to predict the correct order of the current example image
        predictor = Predictor()
        prediction = predictor.make_prediction(img_name)
        # Example images are all shuffled in the "3120" order
        print(prediction)

        # # Visualize the image
        # pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
        # # Example images are all shuffled in the "3120" order
        # final_image = Image.fromarray(np.vstack((np.hstack((pieces[3],pieces[1])),np.hstack((pieces[2],pieces[0])))))
        # final_image.show()