'''
For comparing pure Tensorflow top and Kearas top on a Keras VGG16 pre-trained backbone. 
Also, shows a way to mix Keras with Tensorflow. 

Based on 
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

def  keras_top():
  """
  Uses Kearas top on VGG16  bottom 
  """
  
  from keras import applications
  from keras.preprocessing.image import ImageDataGenerator
  from keras import optimizers
  from keras.models import Model
  from keras.layers import Input, Dropout, Flatten, Dense
  
  # dimensions of our images.
  img_width, img_height = 150, 150
  
  train_data_dir = 'cats_and_dogs_small/train'
  validation_data_dir = 'cats_and_dogs_small/validation'
  nb_train_samples = 2000
  nb_validation_samples = 800
  epochs = 3
  batch_size = 16
  
  # build the VGG16 network
  inputs = Input(shape = (img_width,img_height,3))
  vgg16 = applications.VGG16(weights='imagenet', include_top=False)
  x = vgg16(inputs)
  print('Model loaded.')
  
  # build a classifier model to put on top of the convolutional model
  x = Flatten()(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.5)(x)
  predictions = Dense(1, activation='sigmoid')(x)
  
  
  # note that it is necessary to start with a fully-trained
  # classifier, including the top classifier,
  # in order to successfully do fine-tuning
  #top_model.load_weights(top_model_weights_path)
  
  # add the model on top of the convolutional base
  #model.add(top_model)
  
  # set the first 25 layers (up to the last conv block)
  # to non-trainable (weights will not be updated)
  for layer in vgg16.layers[:25]:
      layer.trainable = False
  
  # compile the model with a SGD/momentum optimizer
  # and a very slow learning rate.
  model = Model(inputs=inputs, outputs=predictions)
  
  model.compile(loss='binary_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])
  
  # prepare data augmentation configuration
  train_datagen = ImageDataGenerator(
      rescale=1. / 255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)
  
  test_datagen = ImageDataGenerator(rescale=1. / 255)
  
  train_generator = train_datagen.flow_from_directory(
      train_data_dir,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='binary')
  
  validation_generator = test_datagen.flow_from_directory(
      validation_data_dir,
      target_size=(img_height, img_width),
      batch_size=batch_size,
      class_mode='binary')
  
  # fine-tune the model
  model.fit_generator(
      train_generator,
      samples_per_epoch=nb_train_samples,
      epochs=epochs,
      validation_data=validation_generator,
      nb_val_samples=nb_validation_samples)
  

def  tf_top():
  """
  Uses Tensorflow top.
  """
  
  from keras.preprocessing.image import ImageDataGenerator
  import tensorflow as tf
  import time
  
  
  def debugPrint(msg, tnsr, summarize=8):
    tnsr = tf.Print(tnsr, [tnsr, tf.shape(tnsr)],  summarize=summarize, message='tf '+msg, name='debugPrint_'+msg.split()[0])
    return tnsr
  
  def dump_variables ():
    
    print ('dump_variables ===============')
    trainable_variables = tf.trainable_variables()
    print ( 'dump_variables', 'trainable_variables', len(trainable_variables), )
    for var in  trainable_variables :
      print (var)
      
    global_variables = tf.global_variables()
    print ( 'dump_variables', 'global_variables', len(global_variables), )
    for var in  global_variables :
      print (var)
      
    local_variables = tf.local_variables()
    print ( 'dump_variables', 'local_variables', len(local_variables), )
    for var in  local_variables :
      print (var)
      
  class FeatureGenerator ():
    # build the VGG16 network
      
    def __init__(self, session=None):
      import keras.applications.vgg16 as  NN
      from keras.models import Model
      from keras import backend as K
      if session :
        K.set_session(session)
      nn1 = NN.VGG16(weights='imagenet', include_top=False, pooling= None)
      for layer in nn1.layers:
        layer.trainable = False
      nn1.name='vgg16-3'
      nn2 = Model (input=nn1.input, outputs=nn1.layers[-2].output)
      self.nn = nn2
      self.nn.summary()
      return  
    
    def get_features (self,  input_image=None):
      features = self.nn.predict(input_image)
      return features
  
  def network ( summary_writer=None, features_placeholder=None, labels_placeholder=None):
    # build a classifier model to put on top of the convolutional model
    #tf.summary.histogram('features_placeholder', features_placeholder)
    x1 = tf.layers.Flatten()(features_placeholder)
    #x1 = features_placeholder
    x2 = tf.layers.Dense(256, activation=tf.nn.relu, name='mydense1')(x1)
    x3 = tf.layers.Dropout(0.5, name='mydropout')(x2) # these  layers are usually not present during prediction
    predictions = tf.layers.Dense(1, activation=tf.nn.sigmoid, name='mydense2')(x3)
    tf.summary.histogram('predictions', predictions)
    
    labels_placeholder = tf.cast ( tf.reshape (labels_placeholder, (-1,1)) , tf.float32)
    tf.summary.histogram('labels_placeholder', labels_placeholder)
    
    # keras binary cross_entropy equivalent
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=labels_placeholder))
    
    # keras binary_accuracy equivalent
    acc = tf.reduce_mean( tf.cast(tf.equal(tf.round(predictions), labels_placeholder), dtype=tf.float32) )
    
    return predictions, loss, acc
    
  
  def train_dataset_batch() :
  
      summary_writer = tf.summary.FileWriter('logs')
      summary_writer_train = tf.summary.FileWriter('logs/train')
      summary_writer_test   = tf.summary.FileWriter('logs/val')
      
      # dimensions of our images.
      img_width, img_height = 150, 150
      
      train_data_dir = 'cats_and_dogs_small/train'
      validation_data_dir = 'cats_and_dogs_small/validation'
      batch_size = 16
      epoch_size = 1000
      epoch_limit = 1000
      batches_per_epoch = epoch_size//batch_size
      
      features_placeholder = tf.placeholder(tf.float32, shape=( None, img_height*3//50, img_width*3//50, 512), name='features')
      labels_placeholder = tf.placeholder(tf.int32, shape=( None,), name='labels')
      pred, loss, acc = network(summary_writer=summary_writer, features_placeholder=features_placeholder, labels_placeholder=labels_placeholder)
      tf.summary.scalar('loss',  loss)
      tf.summary.scalar('acc',  acc)
      loss = tf.reshape(loss, [])
      
      optimizer = tf.train.MomentumOptimizer(1e-4, 0.9)
      gvs = optimizer.compute_gradients(loss)
      train_op = optimizer.apply_gradients(gvs)
    
      init = tf.global_variables_initializer()
      config = tf.ConfigProto(intra_op_parallelism_threads=1)
      sess= tf.Session (config=config)
      fg = FeatureGenerator(sess)
      summaries_op = tf.summary.merge_all()
      #summary_writer.add_graph(sess.graph)
      sess.run(init)
      
      #dump_variables()
      
      # note that it is necessary to start with a fully-trained
      # classifier, including the top classifier,
      # in order to successfully do fine-tuning
      
      # prepare data augmentation configuration
      train_datagen = ImageDataGenerator(
          rescale=1. / 255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)
      
      test_datagen = ImageDataGenerator(rescale=1. / 255)
      
      train_generator = train_datagen.flow_from_directory(
          train_data_dir,
          target_size=(img_height, img_width),
          batch_size=batch_size,
          class_mode='binary')
      
      validation_generator = test_datagen.flow_from_directory(
          validation_data_dir,
          target_size=(img_height, img_width),
          batch_size=batch_size,
          class_mode='binary')
      
      saver = tf.train.Saver()
      
      starttime = time.time()
      epochtime = starttime
      epoch_idx = 0
      batch_idx_train = 0
      batch_idx_val   = 0
      
      for epoch_idx in range (epoch_limit ): 
        curtime = time.time()
        print ('=== epoch', epoch_idx, 'epoch time %.4f'%(curtime - epochtime), 'total time %.4f'%(curtime  - starttime),)
        epochtime = curtime
        

        batchtime = time.time()
        loss_total = 0
        acc_total  = 0
        iter_idx = 0
        loss_t = 0
        acc_t  = 0
        for x, y in train_generator :
          iter_idx +=1
          curtime = time.time()
          print ('train     ', 'batch %d/%d'%(iter_idx,batches_per_epoch ),  
                 'time %.4f'%(curtime - batchtime), 'loss %.4f'%(loss_total/iter_idx), 
                 'acc %.4f'%(acc_total/iter_idx),  '\r', end=" ")
          batchtime = curtime
          featuresnp  = fg.get_features(input_image=x)
          fd = {features_placeholder: featuresnp, labels_placeholder: y }
          loss_t, acc_t, train_op_t, summary_str = sess.run([ loss, acc, train_op, summaries_op], feed_dict=fd)
          summary_writer_train.add_summary(summary_str, batch_idx_train)
          summary_writer_train.flush()
          batch_idx_train +=1
          loss_total += loss_t
          acc_total  += acc_t
          if iter_idx==batches_per_epoch :
            break 
        print ('' )
        if epoch_idx%10 == 0 : 
          saver.save (sess, './vgg-top.ckpt', global_step=batch_idx_train )
        
#        batchtime = time.time()
#        loss_total = 0
#        acc_total  = 0
#        iter_idx = 0
#        loss_t = 0
#        acc_t  = 0
#        for vx, vy in validation_generator:
#          featuresnp  = fg.get_features(input_image=vx)
#          fd = {features_placeholder: featuresnp, labels_placeholder: vy }
#          loss_v, acc_v, summary_str = sess.run([ loss, acc,summaries_op], feed_dict=fd)
#          summary_writer_test.add_summary(summary_str, batch_idx_val)
#          summary_writer_train.flush()
#          batch_idx_val += 1
#          loss_total += loss_v
#          acc_total  += acc_v
#          iter_idx += 1
#          curtime = time.time()
#          print ('validation', 'batch %d/%d'%(iter_idx, batches_per_epoch), 
#                 'time %.4f'%(curtime - batchtime), 'loss %.4f'%(loss_total/iter_idx),  
#                 'acc %.4f'%(acc_total/iter_idx), '\r', end=' ')
#          batchtime = curtime
#          if iter_idx == batches_per_epoch :
#            break
#        print ('')


          
  train_dataset_batch()
        
        
if __name__ == '__main__' :
  import argparse 
  parser = argparse.ArgumentParser(description='Train a Keras and/or Tensorflow top on a pre-trained Keras VGG16 backbone' )
  parser.add_argument('--keras',    help='train  the keras top code ', action='store_true')
  parser.add_argument('--tf',    help='train  the tensorflow  top code ', action='store_true')
  
  args = parser.parse_args()
  
  if args.keras:
    keras_top()
  if args.tf :
    tf_top()
    