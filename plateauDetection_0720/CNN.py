from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

def autoencoding(train_x, test_x, img_dim=128, encoding_dim=32):
  input_img = Input(shape=(img_dim**2,))
  encoded = Dense(encoding_dim, activation='relu')(input_img)
  decoded = Dense(img_dim**2, activation='sigmoid')(encoded)
  autoencoder = Model(input_img, decoded)
  encoder = Model(input_img, encoded)
  encoded_input = Input(shape=(encoding_dim,))
  decoder_layer = autoencoder.layers[-1]
  decoder = Model(encoded_input, decoder_layer(encoded_input))
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

  r_train_x = train_x.astype('float32')/255
  r_train_x = r_train_x.reshape((len(r_train_x), np.prod(r_train_x.shape[1:])))
   
  r_test_x = test_x.astype('float32')/255
  r_test_x = r_test_x.reshape((len(r_test_x), np.prod(r_test_x.shape[1:])))

  autoencoder.fit(r_train_x, r_train_x, epochs=100, batch_size=100, shuffle=True)
  encoded_imgs = encoder.predict(r_test_x)
  decoded_imgs = decoder.predict(encoded_imgs)
  return decoded_imgs



def autoencoding_cnn(train_x, test_x, img_dim=128, encoding_dim=32):
  input_img = Input(shape=(img_dim, img_dim, 1))
  x = Conv2D(32, 3, 3, activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(32, 3, 3, activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)

  # at this point the representation is (7, 7, 32)

  x = Conv2D(32, 3, 3, activation='relu', padding='same')(encoded)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  decoded = Conv2D(1, 3, 3, activation='sigmoid', padding='same')(x)

  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

  r_train_x = np.array(train_x).astype('float32')/255
  r_train_x = np.reshape(r_train_x, (len(r_train_x), img_dim, img_dim, 1))

  r_test_x = np.array(test_x).astype('float32')/255
  r_test_x = np.reshape(r_test_x, (len(r_test_x), img_dim, img_dim, 1))

  autoencoder.fit(r_train_x, r_train_x, epochs=100, batch_size=100, shuffle=True)
  decoded_imgs = autoencoder.predict(r_test_x)
  return decoded_imgs

def make_cnn_X(imgset):
    result=[]
    for idx in range(len(imgset)):
        img=imgset[idx]
        result_img=np.array(img).astype('float32')/255
        result_img=np.reshape(result_img,(result_img.shape[0],result_img.shape[1],1))
        result.append(result_img)
    return np.array(result)

def make_cnn_X_all(imgdict):
    result=[]
    for setidx in range(len(imgdict)):
        imgset=imgdict[setidx]
        for idx in range(len(imgset)):
            img=imgset[idx]
            result_img=np.array(img).astype('float32')/255
            result_img=np.reshape(result_img,(result_img.shape[0],result_img.shape[1],1))
            result.append(result_img)
    return np.array(result)

def train_encoder(model,img):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(img, img, epochs=20, batch_size=1000, verbose=1, shuffle=True)

    return model

def generate_cnn_autoencoder(x_range=900,y_range=110,encoded_dim=32):
    input_img = Input(shape=(y_range, x_range, 1))
    x = Conv2D(10, 5, activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Dense(encoded_dim)(x)
    
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(1, 5, activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)

    return autoencoder
    
def result_imgdict(model,imgdict):
    for imgidx in range(len(imgdict)):
        imgset=imgdict[imgidx]
        predict_img(model,imgdict,dir=imgidx)

def predict_img(model,imgset,dir="result"):
    preds=model.predict(imgset.reshape(imgset.shape[0],imgset.shape[1],imgset.shape[2],1))
    preds=np.uint64(preds*255)
    for pred_idx in range(len(preds)):
        pred=preds[pred_idx]
        result=np.reshape(pred,(pred.shape[0],pred.shape[1]))
        plt.clf()
        plt.imshow(result,origin="lower")
        plt.savefig("./result/autoencoder/"+str(dir)+"_"+str(pred_idx))

def save_imgs(imgset):
    for img_idx in range(len(imgset)):
        img=imgset[img_idx]
        plt.clf()
        plt.imshow(img,origin="lower")
        plt.savefig("./result/autoencoder/35_1_original/"+str(img_idx))