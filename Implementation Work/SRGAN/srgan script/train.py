from Network import Generator, Discriminator

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
from skimage.transform import rescale, resize
from scipy.misc import imresize
import os
from matplotlib.pyplot import imread
import h5py
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import sys

np.random.seed(10)
image_shape = (512,512,1)


# def vgg_loss(y_true, y_pred):
    
#     vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
#     vgg19.trainable = False
#     for l in vgg19.layers:
#         l.trainable = False
#     loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
#     loss_model.trainable = False
#     return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=["binary_crossentropy", "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


## Check for multiple directories and nested directories

def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories

## Load images from directories 

def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                image = h5py.File(os.path.join(d,f), 'r')
                files.append(image)
                file_names.append(os.path.join(d,f))
                count = count + 1
    return files        
                        
## Define Data Loader 

def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext)
    return files


files = load_data("./data/", ".jpg")
x_train_h5 = files[:500]
x_test_h5 = files[600:900]

x_train = [np.expand_dims(array(mat['cjdata']['image'][()]), axis=2) for mat in x_train_h5]
x_test = [np.expand_dims(array(mat['cjdata']['image'][()]), axis=2) for mat in x_test_h5]

print("data loaded")


def hr_images(images):
    images_hr = array(images)
    return images_hr

def lr_images(images_real , downscale):
    
    images = []
    for img in  range(len(images_real)):
        im = cv2.resize(images_real[img], (images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale), interpolation=cv2.INTER_CUBIC)
        images.append(np.expand_dims(im,axis=2))
    images_lr = array(images)
    return images_lr

def preprocess_HR(x):
    return np.divide(x.astype(np.float32), 127.5) - np.ones_like(x,dtype=np.float32)


def deprocess_HR(x):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 


def preprocess_LR(x):
    return np.divide(x.astype(np.float32), 255.)


def deprocess_LR(x):
    x = np.clip(x*255, 0, 255)
    return x

def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 

def deprocess_LRS(x):
    x = np.clip(x*255, 0, 255)
    return x.astype(np.uint8)

x_train_hr = hr_images(x_train)
x_train_hr = normalize(x_train_hr)

x_train_lr = lr_images(x_train, 4)
x_train_lr = normalize(x_train_lr)


x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(x_test, 4)
x_test_lr = normalize(x_test_lr)

## remove unwanted image sizes form train set HR

x_train_hr_new = []
for i in range(len(x_train_hr)):
    if x_train_hr[i].shape == (512,512,1):
        x_train_hr_new.append(x_train_hr[i])
x_train_hr_new = array(x_train_hr_new)

## remove unwanted image sizes form train set LR

x_train_lr_new = []
for idx, img in enumerate(x_train_lr):
    if img.shape == (128,128,1):
        x_train_lr_new.append(img)
x_train_lr_new = array(x_train_lr_new)

## Remove unwanted image sizes from test set HR

x_test_hr_new = []
for i in range(len(x_test_hr)):
    if x_test_hr[i].shape == (512,512,1):
        x_test_hr_new.append(x_test_hr[i])
x_test_hr_new = array(x_test_hr_new)

## Remove unwanted image sizes from test set LR

x_test_lr_new = []
for i in range(len(x_test_lr)):
    if x_test_lr[i].shape == (128,128,1):
        x_test_lr_new.append(x_test_lr[i])
x_test_lr_new = array(x_test_lr_new)

print("data processed")


## Function to compute evaluation metrices

def compare_images(original, generated):
    '''
    Input:
        original : Original HR Image
        generated : Image Generated by GAN
    return:
        returns psnr and ssim metrics
    '''
    scores = []
    scores.append(cv2.PSNR(original, generated))
    scores.append(ssim(original, generated, multichannel=True))
    scores.append(mse(original, generated))
    
    return scores


## Plot and save images after every 20 epochs

def plot_generated_images(epoch,generator, examples=3 , dim=(1, 3), figsize=(15, 5)):
    
    rand_nums = np.random.randint(0, x_test_hr_new.shape[0], size=examples)
    image_batch_hr = denormalize(x_test_hr_new[rand_nums])
    image_batch_lr = np.stack(x_test_lr_new[rand_nums], axis=0)
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    
    if epoch % 20 == 0:
        plt.figure(figsize=figsize)

        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[1].squeeze(), interpolation='nearest')
        plt.title('Down Sampled Image after eopch {}'.format(epoch))
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[1].squeeze(), interpolation='nearest')
        plt.title('Image Generated by Generator of GAN eopch {}'.format(epoch))
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[1].squeeze(), interpolation='nearest')
        plt.title('Original HR Image eopch {}'.format(epoch))
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('./output/gan_generated_image_epoch_%d.png' % epoch)
    
    ## Printing PSNR AND SSIM
    metric_score = compare_images(image_batch_hr[1], generated_image[1])
    PSNR, SSIM, PMSE = metric_score[0], metric_score[1], metric_score[2]
    return (PSNR, SSIM, PMSE)

## Hold the loss valuesand evaluation metrics for each epochs in the lists

hr_loss, lr_loss, gan_loss = [], [], []
m1, m2, m3 = [], [], []

def train(epochs=1, batch_size=128):

    downscale_factor = 4
    
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    
    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss="binary_crossentropy", optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)
    
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, 3)
    gan = get_gan_network(discriminator, shape, generator, adam)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batch_count):
            
            rand_nums = np.random.randint(0, x_train_hr_new.shape[0], size=batch_size)

        image_batch_hr =  np.stack(x_train_hr_new[rand_nums], axis=0)
        image_batch_lr = np.stack(x_train_lr_new[rand_nums], axis=0)
            
        generated_images_sr = generator.predict(image_batch_lr)

        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
        fake_data_Y = np.random.random_sample(batch_size)*0.2

        discriminator.trainable = True

        d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
        d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
        #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        rand_nums = np.random.randint(0, x_train_hr_new.shape[0], size=batch_size)
    
        image_batch_hr =  np.stack(x_train_hr_new[rand_nums], axis=0)
        image_batch_lr = np.stack(x_train_lr_new[rand_nums], axis=0)

        gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2

        discriminator.trainable = False
        loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])

    sys.stdout.write("Loss HR , Loss LR, Loss GAN")
    print(d_loss_real, d_loss_fake, loss_gan)
    hr_loss.append(d_loss_real)
    lr_loss.append(d_loss_fake)
    gan_loss.append(loss_gan)
    
    PSNR, SSIM, PMSE = plot_generated_images(e, generator)
    m1.append(PSNR)
    m2.append(SSIM)
    m3.append(PMSE)
#     plot_generated_images(e, generator)
    
    if e % 220 == 0:
        generator.save('./output/gen_model%d.h5' % e)
        discriminator.save('./output/dis_model%d.h5' % e)
        gan.save('./output/gan_model%d.h5' % e)


plt.style.use('ggplot')

def visualize_model():

    ## Visualizing
    ## (GAN Loss) Perceptual Loss
    ## (HR Loss) Adversarial Loss
    ## (LR Loss) Content(Reconstruction) loss

    epochs = np.arange(1,len(hr_loss)+1)

    fig = plt.figure(figsize=(10,10))
    plt.subplot(3, 2, 1)

    ## Plotting HR Loss

    plt.plot(epochs, hr_loss, label='hr_loss')
    plt.title('HR Image Loss (Adversarial Loss)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.legend()
    plt.savefig('hr_image_loss.png')

    ## Plotting LR Loss

    plt.subplot(3,2,2)
    plt.plot(epochs, lr_loss, label='lr_loss', color='b')
    plt.title('LR Image Loss (Content(Reconstruction) loss)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.legend() 
    plt.savefig('lr_image_loss.png')

    ## Plotting GAN Loss

    loss, fun_1_loss, fun_3_loss = [], [], []

    for l1, l2, l3 in gan_loss:
        loss.append(l1)
        fun_1_loss.append(l2)
        fun_3_loss.append(l3)

    plt.subplot(3,2,3)
    plt.plot(epochs, loss, label='loss', color='g')
    plt.title('GAN Loss (Perceptual Loss)')
    plt.ylabel('loss')
    plt.xlabel('Epoch #')
    plt.legend() 

    fig.tight_layout()
    plt.savefig('gan_loss.png')

    ## Visualize PSNR for each epoch

    plt.plot(epochs, m1, label='psnr', color='c')
    plt.title('PSNR for Each Epoch')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch #')
    plt.legend()
    plt.savefig('psnr_loss.png')


    ## Visualize SSIM for each epoch

    plt.plot(epochs, m2, label='ssim', color='m')
    plt.title('SSIM for Each Epoch')
    plt.xlabel('HR/LR Image Loss')
    plt.ylabel('Epoch #')
    plt.legend()
    plt.savefig('ssim_loss.png')

    ## Visualize MSE for each epoch

    plt.plot(epochs, m3, label='mse')
    plt.title('MSE for Each Epoch')
    plt.ylabel('Mean Square Error')
    plt.xlabel('Epoch #')
    plt.legend()
    plt.savefig('mse_loss.png')

    print('Avg. PSNR and Std for Generated Images:', np.mean(np.array(m1)), np.std(np.array(m1)))
    print('Avg. SSIM and Std for Generated Images:', np.mean(np.array(m2)), np.std(np.array(m2)))
    print('Avg. MSE and Std for Generated Images:', np.mean(np.array(m3)), np.std(np.array(m3)))



train(20000,4)


