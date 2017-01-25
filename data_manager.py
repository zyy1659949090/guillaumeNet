


def load_camvid_train_data(tab): # load les fichiers de tab
    train_data =[]
    train_label = []
    for i in tab:
        train_data.extend(np.load(pathSave+'camvid/train_data_'+str(i)+'.npy'))
        train_label.extend(np.load(pathSave+'camvid/train_label_'+str(i)+'.npy'))
    return np.array(train_data),np.array(train_label)

def normalized(rgb, w=224, h=224):
    rgb_output = cv2.resize(rgb,(w,h))
    #return rgb_output
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm
