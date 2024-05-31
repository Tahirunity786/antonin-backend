import numpy as np

def neighbors(input_matrix,input_array):
    (rows, cols) = input_matrix.shape[:2] # New
    indexRow = input_array[0]
    indexCol = input_array[1]
    output_array = [0] * 4 # New - I like pre-allocating

    # Edit
    output_array[0] = input_matrix[(indexRow - 1) % rows,indexCol]
    output_array[1] = input_matrix[indexRow,(indexCol + 1) % cols]
    output_array[2] = input_matrix[(indexRow + 1) % rows,indexCol]
    output_array[3] = input_matrix[indexRow,(indexCol - 1) % cols]
    return output_array

def bwmorph_remove(input_matrix):
    output_matrix = input_matrix.copy()
    # Change. Ensure single channel
    if len(output_matrix.shape) == 3:
        output_matrix = output_matrix[:, :, 0]
    nRows,nCols = output_matrix.shape # Change
    orig = output_matrix.copy() # Need another one for checking
    for indexRow in range(0,nRows):
        for indexCol in range(0,nCols):
            center_pixel = [indexRow,indexCol]
            neighbor_array = neighbors(orig, center_pixel) # Change to use unmodified image
            if np.all(neighbor_array): # Change
                output_matrix[indexRow,indexCol] = 0

    return output_matrix