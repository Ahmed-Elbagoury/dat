from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Concatenate, Conv2D, MaxPooling2D, Flatten
import random
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import math
from collections import defaultdict
import argparse
import sys

batch_size = 512
num_classes = 10
pic_size = 784
args = None  # command line arguments

class View(object):
    '''One network'''
    view_id = 0
    def __init__(self):
        View.view_id += 1
        self.view_id = View.view_id
        inputs = Input(shape=(28, 28, 1))
        x = Conv2D(16, (3, 3), padding='same', activation='relu', name="v%d-1" % self.view_id)(inputs)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        self.disc_layer = Flatten()(x)  # input to the discriminator
        x = Dense(num_classes, activation='softmax', name="v%d" % self.view_id)(self.disc_layer)
        self.gen_layer = x  # output of the generator
        output_layer = x

        self.model = Model(inputs=inputs, outputs=output_layer)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def predict_proba(self, x):
        return self.model.predict(x)


def generate_training_data(x_train, x_label):
    '''Given n training examples, return 2*n'''
    x_1 = []
    x_2 = []
    y_1 = []
    y_2 = []
    is_same = []
    n = len(x_train)

    x_1.extend(x_train)
    x_2.extend(x_train)
    y_1.extend(x_label)
    y_2.extend(x_label)
    is_same.extend(np.ones(n))  # training instances of view1 is_same to view2

    shuffle_index = random.sample(range(n), n)
    x_1.extend(x_train)
    x_2.extend(x_train[shuffle_index])
    y_1.extend(x_label)
    y_2.extend(x_label[shuffle_index])
    is_same.extend(np.zeros(n))   # training instances of view1 are different from view2
    return np.array(x_1), np.array(x_2), np.array(y_1), np.array(y_2), np.array(is_same)


def simple_ensemble(clf1, clf2, x):
    '''returns accuracy of first classifier, second classifier, ensemble'''
    y1_proba = clf1.predict_proba(x)
    y2_proba = clf2.predict_proba(x)
    r1 = np.argmax(y1_proba, axis=1)
    r2 = np.argmax(y2_proba, axis=1)
    r3 = np.argmax(y1_proba + y2_proba, axis=1)
    return r1, r2, r3


# Train the generators and the discriminator models on mnist data.
def train_mnist():

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # views
    view1 = View()  
    view2 = View()
    gen, disc = build_model(view1, view2)  # the generator consists mainly of view1 and view2
    gen_initial_weights = gen.get_weights()  # used below to reinitialize weights between runs

    n = len(y_train)
    for number_of_labeled_exampled in [args.labeled]: 
        number_of_labeled_exampled = min(n, number_of_labeled_exampled)
        labeled_index = random.sample(range(n), number_of_labeled_exampled)
        gen.set_weights(gen_initial_weights)  # reset weights
        train(view1, view2, gen, disc, x_train[labeled_index], y_train[labeled_index], x_test, y_test)


def build_model(view1, view2):
    '''Build a model that encompasses the discriminator and generator.
    The generator consists mainly of view1 and view2.
    When 'baseline' is set, the two views are trained independently.'''

    # Discriminator
    v1 = Dense(16, activation='relu', name='d_v1')(view1.disc_layer)
    v2 = Dense(16, activation='relu', name='d_v2')(view2.disc_layer)
    x = keras.layers.concatenate([v1, v2])
    x = Dense(1, activation='sigmoid', name='disc')(x)
    disc_loss = 'binary_crossentropy'
    disc = Model(inputs=[view1.model.input, view2.model.input], outputs=x)

    if not args.baseline:
        # Generator contains both views + discriminator.
        gen = Model(inputs=[view1.model.input, view2.model.input],
                    outputs=[view1.gen_layer, view2.gen_layer, disc.output])
    else:
        # Generator: the baseline does not use the discriminator
        # output as part of the loss function
        gen = Model(inputs=[view1.model.input, view2.model.input],
                    outputs=[view1.gen_layer, view2.gen_layer])

    # collect the layers of each part of the network
    input_layers = set([l for l in gen.layers if type(l) == keras.engine.topology.InputLayer])
    disc_layers = set(disc.layers) - (set(view1.model.layers) | set(view2.model.layers)) - input_layers
    gen_layers = set(gen.layers) - disc_layers - input_layers

    input_layers = list(input_layers)
    disc_layers = list(disc_layers)
    gen_layers = list(gen_layers)
    def set_trainable(layers, is_trainable):
        for l in layers:
            l.trainable = is_trainable

    # generator not trainable, discriminator is trainable, then compile
    # changing trainablility after compile doesn't have any effect
    set_trainable(gen_layers, False)
    set_trainable(disc_layers, True)

    disc.compile(loss=disc_loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    if not args.experiment:
        disc.summary()  # the summary should show that only the discriminator parameters are trainable

    # generator trainable, discriminator is not, then compile
    set_trainable(gen_layers, True)
    set_trainable(disc_layers, False)

    metric = 'accuracy'
    loss = 'sparse_categorical_crossentropy'
    if not args.baseline:
        gen.compile(loss=[loss, loss, disc_loss],
                    optimizer='adam',
                    metrics=[metric, 'accuracy'])
    else:
        gen.compile(loss=[loss, loss],
                    optimizer='adam',
                    metrics=[metric])

    if not args.experiment:
        gen.summary()

    return gen, disc


def train(view1, view2, gen, disc, x_train, y_train, x_test, y_test):
    '''Train diverse view1 and view2. Training procedure is inspired by how GANs are trained'''

    (x_1, x_2, y_1, y_2, is_same) = generate_training_data(x_train, y_train)
    x_disc = [x_1, x_2]
    y_disc = [is_same]

    x_gen = [x_1, x_2]
    x_gen_test = [x_test, x_test]

    if not args.baseline:
        y_gen = [y_1, y_2, 1 - is_same]  # The `1 - is_same` is very important. This is to maximize discriminator loss.
        y_gen_test = [y_test, y_test, np.ones(len(y_test))]
    else:
        y_gen = [y_1, y_2]  # in the baseline model, the two views are trained independently.
        y_gen_test = [y_test, y_test]

    def one_batch(data, index):
        batch = []
        for col in data:
            batch.append(col[index])
        return batch

    n = len(x_1)
    all_metrics = defaultdict(list)
    batchs_per_epoch = int(math.ceil(max(1, n/float(batch_size))))
    if n < 5000: # for small n, do 100 batchs per epoch
        batchs_per_epoch = 100
    print("datasize: %d, batch_per_epoch: %d" % (n, batchs_per_epoch))

    # training as in GANs, train discriminator on one batch with the generator freezed,
    # then flip; train generator on one batch with the discriminator freezed
    epoch = 0
    for i in range(1, batchs_per_epoch * args.epochs):
        if batch_size > n:
            shuffle_index = range(n)
        else:
            shuffle_index = random.sample(range(n), batch_size)
        x = one_batch(x_disc, shuffle_index)
        y = one_batch(y_disc, shuffle_index)

        batch_metrics = disc.train_on_batch(x, y)
        for j in range(len(disc.metrics_names)):
            all_metrics['d_' + disc.metrics_names[j]].append(batch_metrics[j])

        x = one_batch(x_gen, shuffle_index)
        y = one_batch(y_gen, shuffle_index)
        batch_metrics = gen.train_on_batch(x, y)
        for j in range(len(gen.metrics_names)):
            all_metrics['g_' + gen.metrics_names[j]].append(batch_metrics[j])

        if not args.experiment:
            sys.stdout.write('\r%d %d ' % (epoch, i))
            for metric_name, metric_values in all_metrics.items():
                sys.stdout.write('{}:{:.4f} '.format(metric_name, np.average(metric_values)))
            sys.stdout.flush()

        if i % batchs_per_epoch == 0:
            epoch += 1

            disc_acc = np.average(all_metrics['g_disc_acc'])
            view1_acc = np.average(all_metrics['g_v1_acc'])
            view2_acc =  np.average(all_metrics['g_v2_acc'])
            pred_v1, pred_v2, pred_ensemble = simple_ensemble(view1, view2, x_train)
            acc_v1 = accuracy_score(y_train, pred_v1)
            acc_v2 = accuracy_score(y_train, pred_v2)
            acc_ensemble = accuracy_score(y_train, pred_ensemble)
            print ("\nTrain results view1: %0.4f, view2: %0.4f, v1_acc: %0.4f, v2_acc: %0.4f, ensemble: %0.4f, disc: %0.4f" % (view1_acc, view2_acc, acc_v1, acc_v2, acc_ensemble, disc_acc))

            gen_metrics_test = defaultdict(lambda: float('nan'), zip(gen.metrics_names, gen.evaluate(x_gen_test, y_gen_test)))
            disc_acc = gen_metrics_test['disc_acc']
            view1_acc = gen_metrics_test['v1_acc']
            view2_acc = gen_metrics_test['v2_acc']
            pred_v1, pred_v2, pred_ensemble = simple_ensemble(view1, view2, x_test)
            acc_v1 = accuracy_score(y_test, pred_v1)
            acc_v2 = accuracy_score(y_test, pred_v2)
            acc_ensemble = accuracy_score(y_test, pred_ensemble)
            print ("\n Test results view1: %0.4f, view2: %0.4f, v1_acc: %0.4f, v2_acc: %0.4f, ensemble: %0.4f, disc: %0.4f" % (view1_acc, view2_acc, acc_v1, acc_v2, acc_ensemble, disc_acc))

            # remove metrics of the previous epoch   
            for k, v in all_metrics.items():
                v[:] = []

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-labeled', type=int, default=10**9, help='number of labeled examples')
    parser.add_argument('-baseline', default=0, type=int, help='is baseline')
    parser.add_argument('-experiment', default=False, action='store_const', const=True, help='run experiment not a single run')
    parser.add_argument('-epochs', type=int, default=40, help='number of epochs')
    args = parser.parse_args()
    print("Command line args: %s" % args)

    if not args.experiment:
        seed = args.labeled
        random.seed(seed)
        np.random.seed(seed)
        train_mnist()
        return

    for labeled in [50, 100, 200, 500, 1000]:
        for repeat in [0, 1, 2, 3, 4]:
            seed = labeled + repeat + 1
            for baseline in [0, 1]:
                View.view_id = 0
                random.seed(seed)  # use the same seed to get the same labeled examples
                np.random.seed(seed)
                print('Experiment -- labeled: %d, baseline: %d, repeat: %d, seed: %d' %
                    ( labeled, baseline, repeat, seed))
                args.labeled = labeled
                args.baseline = baseline
                train_mnist()


if __name__ == "__main__":
    main()
