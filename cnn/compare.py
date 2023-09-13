# Pavel, Hamza Reza
#Reference: Based on the tutorial in the following link https://www.tensorflow.org/tutorials/images/cnn

from cnn import CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def test_eval():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    #keras model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    test_loss, test_acc = model.evaluate(test_images,  test_labels)

    #my cnn
    my_cnn = CNN()
    my_cnn.add_input_layer((32, 32, 3), name="inputlayer")
    my_cnn.append_conv2d_layer(num_of_filters=32,kernel_size=3,activation='relu', name='conv1')
    my_cnn.append_maxpooling2d_layer(2, name='mp1')
    my_cnn.append_conv2d_layer(num_of_filters=64,kernel_size=3,activation='relu', name='conv2')
    my_cnn.append_maxpooling2d_layer(2, name='mp2')
    my_cnn.append_conv2d_layer(num_of_filters=64,kernel_size=3,activation='relu', name='conv3')
    my_cnn.append_flatten_layer(name='flatten1')
    my_cnn.append_dense_layer(num_nodes=64,activation='relu',name='dense1')
    my_cnn.append_dense_layer(num_nodes=10,name='dense2')
    my_cnn.set_optimizer('adam')
    my_cnn.set_metric('accuracy')
    my_cnn.set_loss_function('SparseCategoricalCrossentropy')
    myacc = my_cnn.evaluate(test_images,  test_labels)
    assert test_acc == myacc



def test_train():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    #keras model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))
    loss_list = history['loss']

    #my cnn
    my_cnn = CNN()
    my_cnn.add_input_layer((32, 32, 3), name="inputlayer")
    my_cnn.append_conv2d_layer(num_of_filters=32,kernel_size=3,activation='relu', name='conv1')
    my_cnn.append_maxpooling2d_layer(2, name='mp1')
    my_cnn.append_conv2d_layer(num_of_filters=64,kernel_size=3,activation='relu', name='conv2')
    my_cnn.append_maxpooling2d_layer(2, name='mp2')
    my_cnn.append_conv2d_layer(num_of_filters=64,kernel_size=3,activation='relu', name='conv3')
    my_cnn.append_flatten_layer(name='flatten1')
    my_cnn.append_dense_layer(num_nodes=64,activation='relu',name='dense1')
    my_cnn.append_dense_layer(num_nodes=10,name='dense2')
    my_cnn.set_optimizer('adam')
    my_cnn.set_metric('accuracy')
    my_cnn.set_loss_function('SparseCategoricalCrossentropy')
    my_loss_list = my_cnn.train(train_images, train_labels, 100,5)
    assert loss_list == my_loss_list



