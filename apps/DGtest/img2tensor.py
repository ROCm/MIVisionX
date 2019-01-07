import numpy as np
import sys
import getopt
import os.path
import os
from PIL import Image

def createImagelist(imageDir, imageListDir) :
    print('')
    print('Reading png images...')
    os.system('ls -d '+ imageDir + '/*.png >> ' + imageListDir)
    print('')
    print('Reading jpg images...')
    os.system('ls -d '+ imageDir + '/*.jpg >> ' + imageListDir)
    print('')
    print('Reading jpeg images...')
    os.system('ls -d '+ imageDir + '/*.jpeg >> ' + imageListDir)
    return imageListDir

def load_image(fileName) :
	img = Image.open(fileName)
	data = np.asarray(img, dtype=np.float32)
	data = data.flatten()
	return data

if((len(sys.argv) != 7)):
	print('Usage: python img2tensor.py -d <image_directory> -i <imagetag.txt> -o <output_tensor.f32>')
	quit()

opts, args = getopt.getopt(sys.argv[1:], 'd:i:o:')
imageDir = ''
tensorDir = ''
imageListDir = ''

for opt, arg in opts:
    if opt == '-d':
        imageDir = arg
    elif opt == '-i':
		imageListDir = arg
    elif opt == '-o':
		tensorDir = arg

list_filename = createImagelist(imageDir, imageListDir)
with open(list_filename) as f:
	imglist = f.readlines()
imglist = [x.strip() for x in imglist]
#os.system('rm ' + imageListDir)
num_images = len(imglist)

if num_images == 0 :
    print('')
    print('There are no images found in the directory: ' + imageDir)
    quit()

print('')
print('Total number of images read: ' + str(num_images))
print('Creating a tensor with batch size ' + str(num_images) + '...')

# Read images and convert to tensor
op_tensor = np.array([], dtype=np.float32)
for img in imglist:
    rgbBuf = load_image(img)
    op_tensor = np.append(op_tensor, rgbBuf)

with open(tensorDir, 'w') as f:
    op_tensor.astype('float32').tofile(f)

print('')
print('Done')
