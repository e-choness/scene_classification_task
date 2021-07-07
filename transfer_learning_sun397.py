import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tensorflow.keras import Sequential, layers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Input, Dense, BatchNormalization,GlobalAveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
# import tensorflow_addons as tfa

import time
from imageio import imwrite,imread
from vgg import vgg16_hybrid_places_1365 as vgg16_hybrid
from vgg import vgg16_places_365 as vgg16_place

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE

sun397_path = './sun397'
train_partition_path = './sun397/partition/Training_08.txt'
test_partition_path = './sun397/partition/Testing_08.txt'

train_partition = np.loadtxt(train_partition_path, dtype='str', encoding='utf-8')
test_partition = np.loadtxt(test_partition_path, dtype='str', encoding='utf-8')

train_dir = './sun397/train/'
test_dir = './sun397/test/'


def get_sun397():
    length = train_partition.shape[0]
    y_train = []
    y_test = []

    for i in range(0, length, 50):
        trainp = train_partition[i].split('/')
        testp = test_partition[i].split('/')
        train_label = ''
        test_label = ''
        for j in range(1, len(trainp) - 1):
            train_label += trainp[j] + '_'
            test_label += testp[j] + '_'
        y_train.append(train_label[:-1])
        y_test.append(test_label[:-1])
        tag = train_label[:-1]
        label_train = os.path.join(train_dir, tag)
        label_test = os.path.join(test_dir, tag)
        os.makedirs(label_train)
        os.makedirs(label_test)
        for j in range(i, i + 50):
            xtrain_path = sun397_path + train_partition[j]
            xtest_path = sun397_path + test_partition[j]
            trainp = train_partition[j].split('/')
            testp = test_partition[j].split('/')
            print(xtrain_path, xtest_path, train_label, test_label)
            xtrain = imread(xtrain_path)
            xtest = imread(xtest_path)
            if len(xtrain.shape) == 3 and xtrain.shape[2] == 4:
                xtrain = xtrain[:, :, 0:3]
            elif len(xtrain.shape) == 2 or xtrain.shape[2] == 1:
                xtrain = np.array([[[s, s, s] for s in r] for r in xtrain])
            if len(xtest.shape) == 3 and xtest.shape[2] == 4:
                xtest = xtest[:, :, 0:3]
            elif len(xtest.shape) == 2 or xtest.shape[2] == 1:
                xtest = np.array([[[s, s, s] for s in r] for r in xtest])
            print(xtrain.dtype, xtrain.shape, xtest.dtype, xtest.shape)
            xtrain = np.array(Image.fromarray(np.uint8(xtrain)).resize((224, 224)))
            xtest = np.array(Image.fromarray(np.uint8(xtest)).resize((224, 224)))
            print(xtrain.dtype, xtrain.shape, xtest.dtype, xtest.shape)
            xtrain_path = os.path.join(label_train, trainp[-1])
            xtest_path = os.path.join(label_test, testp[-1])
            imwrite(xtrain_path, xtrain)
            imwrite(xtest_path, xtest)


            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
            np.savetxt('./train_labels.txt', y_train)
            np.savetxt('./test_labels.txt', y_test)

def loadData(train_dir, test_dir, validation_split=0.2, batch_size=32, img_size=(224, 224)):
    train_dir = train_dir
    test_dir = test_dir
    batch_size = batch_size
    img_size = img_size
    validation_dataset = None
    if validation_split > 0 and validation_split < 0.5:
        train_dataset = image_dataset_from_directory(train_dir,
                                                     shuffle=True,
                                                     label_mode='categorical',
                                                     validation_split=validation_split,
                                                     subset='training',
                                                     seed=10,
                                                     batch_size=batch_size,
                                                     image_size=img_size)
        validation_dataset = image_dataset_from_directory(train_dir,
                                                     shuffle=True,
                                                     label_mode='categorical',
                                                     validation_split=validation_split,
                                                     subset='validation',
                                                     seed=10,
                                                     batch_size=batch_size,
                                                     image_size=img_size)
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    else:
        train_dataset = image_dataset_from_directory(train_dir,
                                                     shuffle=True,
                                                     label_mode='categorical',
                                                     seed=10,
                                                     batch_size=batch_size,
                                                     image_size=img_size)

    test_dataset = image_dataset_from_directory(test_dir,
                                                label_mode='categorical',
                                                batch_size=batch_size,
                                                image_size=img_size)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    if validation_dataset:
        return train_dataset, validation_dataset, test_dataset
    else:
        return train_dataset, test_dataset


train_dataset, validation_dataset, test_dataset = loadData(train_dir, test_dir)

print(train_partition.shape, test_partition.shape)
print(train_dataset, type(train_dataset))


# def lr_scheduler(epoch, lr):
#     if epoch < 10:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)
def lr_scheduler(epoch):
    if epoch <= 1:
        return 1e-4 # 0.00001
    elif epoch <= 20:
        return 1e-5
    elif epoch <= 30:
        return 1e-6
    else:
        return 1e-7
    return LR

earlyStopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', min_delta=0.001,
            patience=10, mode='max')
scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)


class TransferMode:
    def __init__(self, train_dataset, test_dataset, validation_dataset=None, img_shape=(224, 224, 3), classes=397, batch_size=32):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.classes = classes
        self.img_shape = img_shape
        self.batch_size = batch_size
        if validation_dataset:
            self.validation_dataset = validation_dataset

    def saveLables(self, dataset, name, label_path='./deep_feature'):
        labels = np.concatenate([y for _, y in dataset], axis=0)
        num_classes = len(np.unique(labels))
        labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes, dtype=int)
        file_path = os.path.join(label_path, name)
        if not os.path.isfile(file_path):
            if not os.path.exists(label_path):
                os.makedirs(label_path)
            np.savetxt(file_path, labels, fmt='%d')
        return labels

    def saveFeatures(self, features, name, save_dir='./deep_feature'):
        file_name = name + '_feature.txt'
        file_path = os.path.join(save_dir, file_name)
        if not os.path.isfile(file_path):
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            print('save features in ' + name)
            np.savetxt(file_path, features)

    def deep_features(self, model, data, feature_layer='feature_batch_average'):
        feature_average = Model(model.input, model.get_layer(feature_layer).output)
        features = feature_average.predict(data, batch_size = self.batch_size)
        return features

    def dataAugment(self, show=True, nums=1):
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        if show:
            for image, _ in self.train_dataset.take(nums):
                plt.figure(figsize=(10, 10))
                first_image = image[0]
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    augmented_image = self.data_augmentation(tf.expand_dims(first_image, 0))
                    plt.imshow(augmented_image[0] / 255)
                    plt.axis('off')
                plt.show()

    def train(self, model, initial_epochs=5, save_dir=None):
        self.initial_epochs = initial_epochs
        loss0, accuracy0 = model.evaluate(self.validation_dataset)

        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        history = model.fit(self.train_dataset,
                            epochs=initial_epochs,
                            validation_data=self.validation_dataset,
                            callbacks=[earlyStopping, scheduler],
                            )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        self.acc = acc
        self.val_acc = val_acc
        self.loss = loss
        self.val_loss = val_loss
        self.loss0 = int(loss0)
        self.history_epoch = history.epoch[-1]
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, self.loss0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

        train_loss, train_accuracy = model.evaluate(self.validation_dataset)

        print("train loss: {:.2f}".format(train_loss))
        print("train accuracy: {:.2f}".format(train_accuracy))

        loss, accuracy = model.evaluate(self.test_dataset)
        print('Test accuracy :', accuracy)
        self.model = model
        if save_dir != None:
            model.save_weights(os.path.join(save_dir, "weight_initial.h5"))

    def finetune(self, model, start=10, epochs=3, save_dir=None):
        weight_dir = os.path.join(save_dir, "weight_initial.h5")
        if not os.path.isfile(weight_dir):
            print("Please run TransferModel.train first")
            return
        model.load_weights(weight_dir)

        print("Number of layers in the base model: ", len(model.layers))
        print("Fine-tune from ", start, ' layer onwards')
        model.trainable = True
        # Freeze all the layers before the start
        for layer in model.layers[:start]:
            layer.trainable = False
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=self.base_learning_rate / 100),
                      metrics=['accuracy'])
        model.summary()
        # print(len(model.trainable_variables))

        history_fine = model.fit(self.train_dataset,
                                 epochs=epochs,
                                 batch_size=self.batch_size,
                                 validation_data=self.validation_dataset,
                                 validation_batch_size=self.batch_size,
                                 callbacks=[earlyStopping],
                                 )



        loss, accuracy = model.evaluate(self.test_dataset)
        print('Test accuracy :', accuracy)
        # self.model = model
        if save_dir != None:
            model.save_weights('weights_finetuned.h5', overwrite=True)
        return model

    def vgg16_place_365(self, pretained=True):
        img_shape = self.img_shape
        weights = 'places' if pretained == True else None
        trainable = False if pretained == True else True
        base_model = vgg16_place.VGG16_Places365(input_shape=img_shape,
                                                    include_top=True,
                                                    # pooling='avg',
                                                    classes=365,
                                                    weights=weights,
                                                    )

        if trainable == False:
            for layer in base_model.layers[:-6]:
                layer.trainable = False
        base_model.summary()

        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        inputs = Input(shape=img_shape)
        x = self.data_augmentation(inputs)
        print('---shape of input: ', inputs.shape)
        x = preprocess_input(x)
        print('---shape of x preprocess: ', x.shape)
        conv_model = Model(base_model.input, base_model.get_layer('block5_conv3').output)
        avg_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="avg_pooling", padding='valid')
        max_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool", padding='valid')
        # adaptive_max_layer = tfa.layers.AdaptiveMaxPooling2D((2, 2))
        # feature_concat_layer = layers.Concatenate(axis=1)([adaptive_avg_layer, adaptive_max_layer])

        # feature_model = Model(base_model.input, base_model.get_layer('flatten').output)
        fc1 = base_model.get_layer('fc1')
        fc2 = base_model.get_layer('fc2')
        flatten = base_model.get_layer('flatten')
        x = conv_model(x)
        prev = x
        # print('prev: ', x)
        x_avg = avg_layer(prev)
        x_max = max_layer(prev)
        x = layers.Add()([x_avg, x_max])
        # print('adaptive avg layer: ', x_avg)
        # print('adaptive max layer: ', x_max)
        # print('concatenate: ', x.shape)
        x = flatten(x)
        x = fc1(x)
        prev = x
        x = fc2(x)
        # x = layers.Add(name='feature_fusion')([prev, x])

        # Classification block
        # x = Model(base_model.input, base_model.get_layer('fc2').output)(x)

        outputs = Dense(self.classes, activation='softmax', name='prediction')(x)

        print('---shape of  prediction: ', outputs.shape)
        model = Model(inputs, outputs, name='vgg16_place')
        self.base_learning_rate = 0.0001 if pretained else 0.001
        model.compile(optimizer=RMSprop(lr=self.base_learning_rate, clipvalue=1.0, decay=1e-8),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        model.summary()
        return model

    def vgg16_hybrid_place_1365(self, pretained=True):
        img_shape = self.img_shape
        weights = 'places' if pretained == True else None
        trainable = False if pretained == True else True
        base_model = vgg16_hybrid.VGG16_Hybrid_1365(input_shape=img_shape,
                                                    include_top=True,
                                                    # pooling='avg',
                                                    classes=1365,
                                                    weights=weights,
                                                    )

        if trainable == False:
            for layer in base_model.layers[:-6]:
                layer.trainable = False
        base_model.summary()

        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        inputs = Input(shape=img_shape)
        x = self.data_augmentation(inputs)
        print('---shape of input: ', inputs.shape)
        x = preprocess_input(x)
        print('---shape of x preprocess: ', x.shape)
        feature_model = Model(base_model.input, base_model.get_layer('flatten').output)
        fc1 = base_model.get_layer('fc1')
        fc2 = base_model.get_layer('fc2')
        x = feature_model(x)
        x = fc1(x)
        prev = x
        x = fc2(x)
        feature_fused = layers.Add(name='feature_fusion')([prev, x])

        # Classification block
        # x = base_model(x)
        # x = Flatten(name='feature_batch_average')(x)
        # x = Dense(4096, activation='relu', name='fc1')(x)
        # x = Dropout(0.5, name='drop_fc1')(x)
        # x = Dense(4096, activation='relu', name='fc2')(x)
        # x = Dropout(0.5, name='drop_fc2')(x)
        outputs = Dense(self.classes, activation='softmax', name='prediction')(feature_fused)

        print('---shape of  prediction: ', outputs.shape)
        model = Model(inputs, outputs, name='vgg16_hybrid')
        self.base_learning_rate = 0.0001 if pretained else 0.001
        model.compile(optimizer=RMSprop(lr=self.base_learning_rate, clipvalue=1.0, decay=1e-8),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
        model.summary()
        return model

# save_dir = './vgg_1365'

start = time.clock()
save_dir = './vgg_365'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
T = TransferMode(train_dataset, test_dataset, validation_dataset)
T.dataAugment()
vgg_model = T.vgg16_place_365()
T.train(vgg_model, initial_epochs=13, save_dir=save_dir)
elapsed = time.clock()
elapsed = elapsed - start
print("Time spent in (function name) is: ", elapsed)

# T.finetune(vgg_model, save_dir=save_dir, epochs=3)

# model = DenseNet201(weights="imagenet", include_top=False, classes=397, input_shape=(224,224,3))
# print(model.summary())
#
# x = model.get_layer('relu').output
# x = GlobalAveragePooling2D(name='pool')(x)
# x = Dense(397, activation='softmax', name='fc1')(x)
#
# model_updated = Model(inputs=model.input, outputs=x)
#
# model_updated.save_weights('model_initial.h5')




# model_updated.load_weights('model_initial.h5')
# training_runs = []
# for i in range(3):
#     model_updated.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
#     tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
#     history = model_updated.fit(train_dataset, epochs=7, batch_size=16,
#                                 validation_data=validation_dataset, validation_batch_size=16)
#
#
#     training_runs.append(history)
#
#     model_updated.get_weights()
#     if i == 2:
#         model_updated.save_weights('model1_from_scratch.h5')
#     else:
#         model_updated.load_weights('model_initial.h5')


