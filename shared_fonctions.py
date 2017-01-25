import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def load_camvid_train_data(tab): # load les fichiers de tab
    train_data =[]
    train_label = []
    for i in tab:
        train_data.extend(np.load('data/camvid/train_data_'+str(i)+'.npy'))
        train_label.extend(np.load('data/camvid/train_label_'+str(i)+'.npy'))
    return np.array(train_data),np.array(train_label)

def load_ctscp_data(dset,kind,tab): # load les fichiers de tab
    train_data =[]
    train_label = []
    for city in tab:
	print "Loading ",city
        train_data.extend(np.load('data/'+dset+'/'+kind+'_data_'+city+'.npy'))
        train_label.extend(np.load('data/'+dset+'/'+kind+'_label_'+city+'.npy'))
    return np.array(train_data),np.array(train_label)


def normalized(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img_output

def histoclass(train_label):
    tab = [np.argmax(i) for i in train_label]
    return np.bincount(tab)

def show_sample(data,label, nb, classes_names):
	dataset_size = len(data)
	for i in range(nb):
	    plt.figure(i)
	    j = np.random.randint(dataset_size)
	    plt.imshow(data[j],interpolation='none')
	    plt.title(classes_names[np.argmax(label[j])])
	    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()

def get_cities(kind): # kind = 'train','test' or 'val'
    return os.listdir('cityscape/labels/'+kind);

def load_class(i, nb_max,cities = None):
    a=[]
    if(cities):
        for city in cities:
            a.extend(np.load('data/cityscape/train/'+str(i)+'/'+city+'.npy'))
    else:
        a = np.load('data/cityscape/train/'+str(i)+'/all_cities.npy')
    if nb_max>0:
	np.random.shuffle(a)
	return a[:nb_max]
    return a
	

def create_label_matrix(i,nb,nb_classes):
    res = np.zeros((nb,nb_classes))
    res[:,i]=np.ones(nb)
    return res


