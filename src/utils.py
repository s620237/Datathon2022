import os
import numpy as np

def get_pieces(img, rows, cols, row_cut_size, col_cut_size):
    pieces = []
    for r in range(0, rows, row_cut_size):
        for c in range(0, cols, col_cut_size):
            pieces.append(img[r:r+row_cut_size, c:c+col_cut_size, :])
    return pieces

# Splits an image into uniformly sized puzzle pieces
def get_uniform_rectangular_split(img, puzzle_dim_x, puzzle_dim_y):
    rows = img.shape[0]
    cols = img.shape[1]
    if rows % puzzle_dim_y != 0 or cols % puzzle_dim_x != 0:
        print('Please ensure image dimensions are divisible by desired puzzle dimensions.')
    row_cut_size = rows // puzzle_dim_y
    col_cut_size = cols // puzzle_dim_x

    pieces = get_pieces(img, rows, cols, row_cut_size, col_cut_size)

    return pieces

def getData(path):
  train_folder_path = path

  trainingSubfolders = os.listdir(train_folder_path)
  x = []
  y = []
  for folderLabel in trainingSubfolders:

    testingImages = os.listdir(train_folder_path+'/'+folderLabel)

    for imageTitle in testingImages:
      image = Image.open(train_folder_path+'/'+folderLabel+'/'+imageTitle)
      image = np.array(image).astype('float16')
      image = image / 255 - 0.5

      x.append(get_uniform_rectangular_split(image, 2, 2))

      # x.append(image)
      y.append([int(i) for i in folderLabel])

  data = (np.array(x), np.expand_dims(np.array(y), axis=-1))

  return data