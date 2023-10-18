import struct as st
import numpy as np

filename = {
    'images': 'train-images-idx3-ubyte',
    'lables': 'train-labels-idx1-ubyte'
}

trainImages_file = open(filename['images'], 'rb')
trainImages_file.seek(0)

magic = st.unpack('>4B', trainImages_file.read(4))
print(magic)

nImg = st.unpack('>I', trainImages_file.read(4))[0] #num of images
print(nImg)
nR = st.unpack('>I', trainImages_file.read(4))[0] #num of rows
print(nR)
nC = st.unpack('>I', trainImages_file.read(4))[0] #num of column
print(nC)

nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal, trainImages_file.read(nBytesTotal))).reshape((nImg,nR,nC))
print(images_array[560])