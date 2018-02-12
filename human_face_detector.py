### build a CNN to detect the presence of human faces
import keras
from keras_vggface.vggface import VGGFace # enables loading of the vggface models
from keras.utils import np_utils # not sure I need this, but keeping it for now
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint

# now I want to combine my human and dog images into one dataset
# when I do this I need to make sure there are labels
human_list = human_files[100:]
dog_list = train_files[100:]
dataset = np.append(human_list, dog_list)
labels1 = ['human']*13133
labels2 = ['dog']*(len(dataset)-13133)
labels = labels1+labels2
labels = np.array(labels)
# I haven't added labels at this point, but the first 13,133 files are human and the remainder are not

# now that I have a dataset, I can start developing a CNN to regonize the faces
# I can utilize the xgg-16 because it was trained on faces
# there is a lot of overlap here, and the dataset is small
# so I will slice off the end of the convnet
# add a new fully connected layer with two classes (one for human faces, one for no human face)
# randomize the weights of the new fully connected layer
# freeze the pretrained weights
# train the network to update the weights of the fully connected layer
vggface = VGGFace(model='resnet50', include_top=False, weights='vggface', input_shape=(224, 224, 3))

# setting things up for my problem
nb_class = 2

last_layer = vggface.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
out = Dense(nb_class, activation='softmax', name='classifier')(x)
custom_vgg_model = Model(vggface.input, out)

# Now we have our model built, we can train it
custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])
