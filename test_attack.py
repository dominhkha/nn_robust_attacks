## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            # if inception:
            #     seq = random.sample(range(1,1001), 10)
            # else:
            #     seq = range(data.test_labels.shape[1])
            seq = 1000
            count = 0
            for j in range(0, data.train_data.shape[0]):
                # if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                #     continue
                # print('ok')
                # print(np.argmax(data.test_labels[i]))
                if np.argmax(data.train_labels[j]) == 9.:

                  inputs.append(data.train_data[j])
                  targets.append(np.eye(data.train_labels.shape[1])[7])
                  count += 1
                if count == seq:
                  break
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        # data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        attack = CarliniL2(sess, model, batch_size=100, max_iterations=1000, confidence=0)
        # attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)

        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=0, inception=False)
        print(targets.shape)
        print(inputs.shape)
        # print(data.train_labels.shape)
        # inputs = data.train_data[:1000]
        # targets = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]*1000])
        timestart = time.time()
        adv, ori = attack.attack(np.array(inputs), targets)
        new_adv = []
        new_ori = []
        for index in range(len(adv)):
            if ori[index] is not None and np.sum(ori[index]) != 0:
                new_ori.append(ori[index])
                new_adv.append(adv[index])
        new_ori = np.asarray(new_ori)
        new_adv = np.asarray(new_adv)
        np.save('adv_alexnet_mnist_1kv2.npy', new_adv)
        np.save('ori_alexnet_mnist_1kv2.npy', new_ori)
        print(adv.shape)
        timeend = time.time()

        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
        f = open('alexnet_mnist.txt', 'w')
        text = 'time: ' + str(timeend - timestart)
        if adv is not None:
            text += '\nsuccess_rate: ' + str(adv.shape[0])
        else:
            text += '\nsuccess_rate: 0'
        f.write(text)

        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i])
            print("Adversarial:")
            show(adv[i])
            
            print("Classification:", model.model.predict(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

