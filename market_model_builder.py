from model_builder import AbstractModelBuilder

class MarketPolicyGradientModelBuilder(AbstractModelBuilder):

    def buildModel(self):
        from keras.models import Model
        from keras.layers import concatenate, Conv2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Concatenate , concatenate 
        from keras.layers.advanced_activations import LeakyReLU
        import tensorflow as tf
        import numpy as np
        from tensorflow.keras.models import Sequential 
        from tensorflow.keras.backend import set_image_data_format 
        from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, MaxPool2D 
        from tensorflow.keras import optimizers, losses, utils 

        #os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        #dr_rate = 0.0

        input_shape = [224, 224, 1]
    
        def create_convolution_layers(input_img):
            model = Conv2D(32, (3, 3), padding='same', input_shape=input_shape)(input_img)
            model = LeakyReLU(alpha=0.1)(model)
            model = MaxPool2D((2, 2),padding='same')(model)
            model = Dropout(0.25)(model)
  
            model = Conv2D(64, (3, 3), padding='same')(model)
            model = LeakyReLU(alpha=0.1)(model)
            model = MaxPool2D(pool_size=(2, 2),padding='same')(model)
            model = Dropout(0.25)(model)
    
            model = Conv2D(128, (3, 3), padding='same')(model)
            model = LeakyReLU(alpha=0.1)(model)
            model = MaxPool2D(pool_size=(2, 2),padding='same')(model)
            model = Dropout(0.4)(model)
            
            return model   
    
        img1 = Input(shape=input_shape)
        img1_model = create_convolution_layers(img1)

        img2 = Input(shape=input_shape)
        img2_model = create_convolution_layers(img2)
        
        img3 = Input(shape=input_shape)
        img3_model = create_convolution_layers(img3)

        img4 = Input(shape=input_shape)
        img4_model = create_convolution_layers(img4)
        
        img5 = Input(shape=input_shape)
        img5_model = create_convolution_layers(img5)

        conv = concatenate([img1, img2, img3, img4, img5])

        conv = Flatten()(conv)

        dense = Dense(512)(conv)
        dense = LeakyReLU(alpha=0.1)(dense)
        dense = Dropout(0.5)(dense)

        #output = Dense(num_classes, activation='softmax')(dense)
        output = Dense(2, activation = 'softmax')(dense)
        
        model = Model(inputs=[img1, img2, img3, img4, img5], outputs=[output])
 
    
        return model

class MarketModelBuilder(AbstractModelBuilder):
    
    def buildModel(self):
        from keras.models import Model
        from keras.layers import concatenate, Conv2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Concatenate , concatenate 
        from keras.layers.advanced_activations import LeakyReLU
        import tensorflow as tf
        import numpy as np
        from tensorflow.keras.models import Sequential 
        from tensorflow.keras.backend import set_image_data_format 
        from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, MaxPool2D 
        from tensorflow.keras import optimizers, losses, utils 
        
        dr_rate = 0.0

#        B = Input(shape = (3,))
#        #B = Input(shape=[224, 224, 3])
#        #b = Dense(56, activation = "relu")(B)#
#
#        inputs = []
#        #inputs = [B]
#        #merges = [b]
#        merges = []#

#        for i in range(5):
#            S = Input(shape=[224, 224, 3])
#            
#            inputs.append(S)
#            
#            with tf.device('/cpu:0'):
#                h = Conv2D(28, (3, 3), padding='same', data_format='channels_last')(S)
#                h = LeakyReLU(0.001)(h)
#            with tf.device('/cpu:0'):
#                h = Conv2D(28, (3, 3), padding='same', data_format='channels_last')(h)
#                h = LeakyReLU(0.001)(h)
#                h = MaxPool2D((2,2), padding='same')(h)
#            with tf.device('/cpu:0'):
#                h = Conv2D(56, (3, 3), padding='same', data_format='channels_last')(h)
#                h = LeakyReLU(0.001)(h)
#                #h = MaxPool2D((2,2), padding='same')(h)
#            with tf.device('/cpu:0'):
#                h = Conv2D(56, (3, 3), padding='same', data_format='channels_last')(h)
#                h = LeakyReLU(0.001)(h)
#                h = MaxPool2D((2,2), padding='same')(h)
#            with tf.device('/cpu:0'):
#                h = Conv2D(112, (3, 3), padding='same', data_format='channels_last')(h)
#                h = LeakyReLU(0.001)(h)
#                h = MaxPool2D((2,2), padding='same')(h)
#                
#            h = Flatten()(h)
#            h = Dense(56)(h)
#            h = LeakyReLU(0.001)(h)
#            h = Dropout(dr_rate)(h)
#            #merges.append(h)

           # with tf.device('/cpu:0'):
           #     h = Conv2D(112, (3, 3), padding='same', data_format='channels_last')(S)
           #     h = LeakyReLU(0.001)(h)
           #     h = MaxPool2D((2,2), padding='same')(S)

            #h = Flatten()(h)
            #h = Dense(56)(h)
            #h = LeakyReLU(0.001)(h)
            #h = Dropout(dr_rate)(h)
#            merges.append(h)

#        m = concatenate(merges, axis=1, name = 'concatenate')
        #m = merges
#        m = Dense(112)(m)
#        m = LeakyReLU(0.001)(m)
#        m = Dropout(dr_rate)(m)
#        m = Dense(56)(m)
#        m = LeakyReLU(0.001)(m)
#        m = Dropout(dr_rate)(m)
        
#        V = Dense(2, activation = 'softmax')(m)
#        model = Model(inputs = inputs, outputs = V)
        #model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))

    
        input_shape = [224, 224, 1]
    
        def create_convolution_layers(input_img):
            model = Conv2D(32, (3, 3), padding='same', input_shape=input_shape)(input_img)
            model = LeakyReLU(alpha=0.1)(model)
            model = MaxPool2D((2, 2),padding='same')(model)
            model = Dropout(0.25)(model)
  
            model = Conv2D(64, (3, 3), padding='same')(model)
            model = LeakyReLU(alpha=0.1)(model)
            model = MaxPool2D(pool_size=(2, 2),padding='same')(model)
            model = Dropout(0.25)(model)
    
            model = Conv2D(128, (3, 3), padding='same')(model)
            model = LeakyReLU(alpha=0.1)(model)
            model = MaxPool2D(pool_size=(2, 2),padding='same')(model)
            model = Dropout(0.4)(model)
            
            return model   
    
        img1 = Input(shape=input_shape)
        img1_model = create_convolution_layers(img1)

        img2 = Input(shape=input_shape)
        img2_model = create_convolution_layers(img2)
        
        img3 = Input(shape=input_shape)
        img3_model = create_convolution_layers(img3)

        img4 = Input(shape=input_shape)
        img4_model = create_convolution_layers(img4)
        
        img5 = Input(shape=input_shape)
        img5_model = create_convolution_layers(img5)

        conv = concatenate([img1, img2, img3, img4, img5])

        conv = Flatten()(conv)

        dense = Dense(512)(conv)
        dense = LeakyReLU(alpha=0.1)(dense)
        dense = Dropout(0.5)(dense)

        #output = Dense(num_classes, activation='softmax')(dense)
        output = Dense(2, activation = 'softmax')(dense)
        
        model = Model(inputs=[img1, img2, img3, img4, img5], outputs=[output])
 
    
        return model