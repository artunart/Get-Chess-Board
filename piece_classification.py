import gcb_utils.gcb_utils as gcb_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
from PIL import Image
import tensorflow as tf

class Piece_Classification():
    '''
    A classifier for pieces in individual chess squares with a labeler, trainer and detector. Labeler is implemented in gcb_utils while trainer and detector are implemented in using a simply connected CNN in Tensorflow. Implementation is similar to a basic MNIST classification exercise.
    '''
    
    def __init__(self, model_dir, model_weight_fname=None):
        self.model_dir = model_dir
        self.model_weight_fname = model_weight_fname
        self.model = None    
        if model_weight_fname is None:
            print('No model weight file specified. You can train a new model using Piece_Classification.train or create one using .create_model and then training it.')
        else: 
            self.model = tf.keras.models.load_model(
                      '/'.join([self.model_dir, self.model_weight_fname]))

        self.SQ_LABEL_COLUMNS = ['fname', 'SqColor-BWE', 'PcColor-BWE', 'PcType-PRNBQKE','HumCheck-YN'] 
        self.IMG_FORMATS = ['png', 'jpg', 'jpeg']


    def train(self, train_data_dir, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path,  image_size, model_dir, model_weight_fname, init_weights_dir=None, init_weights_fname=None, epochs=1500, test_data_dir=None,  tf_fit_kwargs=None, tf_compile_kwargs=None, create_model_kwargs=None):
        ''' Trains a model from scratch or from initial weights. The best model in terms of 'val_loss' (if with validation data) or 'loss' (no validation data) is saved. A central storage file containts labeling data for both training and validation (optional) datasets. 
        Args:
        train_data_dir: Absolute/relative path of the directory containing training images
        labels_csv_full_path: absolute/relative full path for the labels csv file (different than the label_dict_file that contains only the class names) 
        label_dict_csv_full_path: absolute/relative full path for the label dict csv (one class per row - no header)
        fname_col: str, csv file column name for filenames
        label_cols: str, csv files column names to concatanate for creating labels. (this is a shortcut, replace with a function based on cols; alternatively feed a df-needs more code)
        image_size: tuple (width, height) in pixels
        model_dir: relative/absolute path for model directory 
        model_weight_fname: filename for weights
        init_weights_dir: relative/absolute path for initial weights to be used in training
        init_weights_fname: filename for initial weights
        epochs: number of epochs for training 
        test_data_dir: relative/absolute path for test images
        tf_fit_kwargs: kwargs dict for tf model fit (tf.keras.Sequential.fit) Refer to keras for more details.
        tf_compile_kwargs: kwargs dict for tf model compile (tf.keras.Sequential.compile). Refer to keras for more details 
        create_model_kwargs: kwaargs dict for some model architecture parameters in self.create_model.
        Returns: None. After any tf printouts, prints the directory where the final weights are output.
        '''
        self.model_dir = model_dir
        self.model_weight_fname = model_weight_fname

        X_train, y_train = self.make_X_y(train_data_dir, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path, image_size)

        if test_data_dir:
            X_test, y_test = self.make_X_y(test_data_dir, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path, image_size)
            callback_save_best = tf.keras.callbacks.ModelCheckpoint('/'.join([model_dir, model_weight_fname]), save_best_only = True, verbose = 2, monitor='val_loss')           
        else:
            X_test, y_test = None, None
            callback_save_best = tf.keras.callbacks.ModelCheckpoint('/'.join([model_dir, model_weight_fname]), save_best_only = True, verbose = 2, monitor='loss')
            

        if init_weights_dir:
            self.model = tf.keras.models.load_model('/'.join([init_weights_dir, init_weights_fname])) 
        else: 
            if not create_model_kwargs:
                create_model_kwargs = {}  
            self.model = self.create_model(label_dict_csv_full_path, image_size, tf_compile_kwargs, **create_model_kwargs)
         
        if not tf_fit_kwargs:
            tf_fit_kwargs = {}
        if 'callbacks' not in tf_fit_kwargs and callback_save_best:
            tf_fit_kwargs['callbacks'] = callback_save_best
        if 'validation_data' not in  tf_fit_kwargs and test_data_dir: 
            tf_fit_kwargs['validation_data'] = (X_test, y_test)
        tf_fit_kwargs['epochs'] = epochs
         
        self.model.fit(X_train, y_train, **tf_fit_kwargs)    
   
        print('Best model saved to: {}'.format('/'.join([model_dir, model_weight_fname])))
    
    
    def predict(self, data_dir, image_size):
        '''Makes classification predictions for the images in data_dir converted the image size indicated.
        Args:
        data_dir: Absolute/relative path of the directory containing prediction images
        image_size: tuple (width, height) in pixels. To convert image size in model specification
        Returns: numpy ndarray. Output of tf.keras.Sequential.predict.
        '''
        if not self.model:
            if not self.model_weight_fname:
                raise Exception('Piece Classification object does not currently have a model. Either initialize with a model or train the model')
            else:
                self.model = tf.keras.models.load_model(
                      '/'.join([self.model_dir, self.model_weight_fname]))
                
        X_list = []

        #Get data files - filter for images, strip extensions
        data_files = [_ for _ in os.listdir(data_dir) if _.split('.')[-1] in self.IMG_FORMATS]
        data_files.sort() 
        print('Prediction Image Count:{0}'.format(len(data_files)))
        
        #Process to create X
        for _fname in data_files:
            _im = Image.open('/'.join([data_dir, _fname]))
            _im = _im.resize(image_size)
            _tmp_array = np.asarray(_im)[:,:,0:3]
            X_list.append(_tmp_array)
            _im.close()
        
        X = np.stack(X_list)
        X = X[:,:,:,0:3].reshape(len(X_list), image_size[0], image_size[1], 3)
   
        return self.model.predict(X) 
    

    def update_labels(self, 
                      main_data_path, 
                      labels_csv_full_path,
                      square_label_columns,
                      update_fn = gcb_utils.square_insert_default_values,
                      update_fn_kwargs ={'label_cols':['SqColor-BWE', 'PcColor-BWE', 'PcType-PRNBQKE'], 'hum_check_col':['HumCheck-YN']}):
       #Passage to label updating GUI.
       if square_label_columns is None:
           square_label_columns  = self.SQUARE_LABEL_COLUMNS
       print('Data path: {}'.format(main_data_path))
       print('Labels csv full path: {}'.format(labels_csv_full_path))
       print('Square label columns: {}'.format(square_label_columns))
       print('Update function: {}'.format(update_fn))
       print('Update function keyword args: {}'.format(update_fn_kwargs))
   
       gcb_utils.insert_data_fnames(main_data_path, labels_csv_full_path, square_label_columns, update_fn = update_fn, update_fn_kwargs=update_fn_kwargs)
       gcb_utils.update_sq_labels(main_data_path, labels_csv_full_path) 


    def create_model(self, label_dict_csv_full_path, image_size, tf_compile_kwargs, conv1_filter_size=15):
        '''Creates and compiles a classier with two convolutional neural network layers to a dense layer which is in turn connected to a dense output layer with softmax activation
        Args:
        labels_csv_full_path: absolute/relative full path for the labels csv file (different than the label_dict_file that contains only the class names) 
        image_size: tuple (width, height) in pixels
        conv1_filter_size: filter size (in pixels)  for first convolution layer. Same value passed for both width and height.
        Returns:
        Creates and compiles the model. Updates self.model.
        '''
        Sequential = tf.keras.Sequential
        Dense = tf.keras.layers.Dense
        Conv2D = tf.keras.layers.Conv2D
        Flatten = tf.keras.layers.Flatten
        MaxPooling2D = tf.keras.layers.MaxPooling2D    
        
        label_list = list(pd.read_csv(label_dict_csv_full_path, header=None).iloc[:,0])
#       print(label_list)
        label_len = len(label_list)
        
        #Model Architecture
        model = Sequential()
        model.add(Conv2D(filters=16,kernel_size=(conv1_filter_size,conv1_filter_size),padding='same',activation='relu',input_shape=(image_size[0], image_size[1],3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        print(model.output_shape)
        model.add(Dense(128, activation='relu'))
        model.add(Dense(label_len, activation='softmax'))

        if not tf_compile_kwargs:
            tf_compile_kwargs = {}
        tf_compile_kwargs['optimizer'] = 'adam'
        tf_compile_kwargs['loss'] = 'categorical_crossentropy'
        tf_compile_kwargs['metrics'] = ['accuracy']
        model.compile(**tf_compile_kwargs)
        return model  


    def make_X_y(self, data_dir, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path, image_size, *args):
        ''' 
        Creates numpy arrays for data (X) and labels (y). Labels numpy arrays indicate the order of string labels in label_dict_csv_full_path.
        Args:
        data_dir: Absolute/relative path of the directory containing training images
        labels_csv_full_path: absolute/relative full path for the labels csv file (different than the label_dict_file that contains only the class names) 
        label_dict_csv_full_path: absolute/relative full path for the label dict csv (one class per row - no header)
        fname_col: str, csv file column name for filenames
        label_cols: list of str, csv files column names to concatanate for creating labels. (this is a shortcut, replace with a function based on cols; alternatively feed a df-needs more code)
        image_size: tuple (width, height) in pixels
        label_dict_csv_full_path: absolute/relative full path for the label dict csv (one class per row - no header)
        Returns: tuple of np.arrays (X, y) where X is data and y is label. 
        '''
        label_list = list(pd.read_csv(label_dict_csv_full_path, header=None).iloc[:,0])
#       print(label_list)
        label_len = len(label_list)
        label_dict = dict(zip(label_list, range(label_len)))
#       print(label_dict)


        X_list = []
        y_list = []
#       print(labels_csv_full_path)
        labels_df = pd.read_csv(labels_csv_full_path)
        labels_df['label'] = ''
        for _ in label_cols:
            labels_df['label'] = labels_df['label'] + labels_df[_].astype('O')
#       print(labels_df['label'].head())

        #Get data files - filter for images, strip extensions
        data_files = [_ for _ in os.listdir(data_dir) if _.split('.')[-1] in self.IMG_FORMATS]
        data_files_no_ext = ['.'.join(_.split('.')[:-1]) for _ in data_files]
#       print(data_files_no_ext)
        data_files_df = pd.DataFrame(data = data_files_no_ext, columns = ['fname_noext'])

        #Strip extensions from filenames in labels_df
        _fname = labels_df['fname'].to_list()
        _fname_noext = ['.'.join(_.split('.')[:-1]) for _ in _fname]
        labels_df['fname_noext'] = _fname_noext
#       print(labels_df['fname_noext'].head())
       
        #Find which files to process - could also make sure again that HumCheck is Y 
        labels_df_toprocess = pd.merge(left=data_files_df, right=labels_df, on='fname_noext')
        print('All image Count:{0}\nData Dir Count:{1}\nFinal Count to Process:{2}'.format(len(labels_df), len(data_files_df), len(labels_df_toprocess))) 
        labels_df_toprocess = labels_df_toprocess.sort_values(by='fname')
        #Process to create X and y
        for _i, _r in labels_df_toprocess.iterrows():
            _fname = _r[fname_col]
            _im = Image.open('/'.join([data_dir, _fname]))
            _im = _im.resize(image_size)
            _tmp_array = np.asarray(_im)[:,:,0:3]
            X_list.append(_tmp_array)
            y_list.append(_r['label'])
            _im.close()

        X = np.stack(X_list)
        X = X[:,:,:,0:3].reshape(len(X_list), image_size[0], image_size[1], 3)
        
        y_np = [label_dict[_] for _ in y_list]
        y_np = np.stack(y_np)
        y = tf.keras.utils.to_categorical(y_np)

#       #check to see if images are ok
#       print(X)
#       print(y)

#       for _i, _X, _y in zip(range(len(X)),X[:5],y_np[:5]):
#           plt.subplot(5,1,_i+1)
#           plt.imshow(_X)
#       plt.gca().set_title([label_list[_] for _ in y_np])
#       plt.show() 

        return X, y
    


    
if __name__ == '__main__':
    model_dir = 'data/model/piece-train/exp1'
    model_weight_fname = 'cnn_pieces.h5' 
    main_source_dir = 'data/raw/squares'
    source_train = 'data/raw/squares/train'
    source_valid = 'data/raw/squares/validation'
    source_test = 'data/raw/squares/test' 
    label_dict_csv_full_path = 'data/model/piece-train/chess_dict_all_EBW.csv'
    labels_csv_full_path = 'data/model/sq_labels.csv'
    fname_col = 'fname'
    label_cols = ['PcColor-BWE', 'PcType-PRNBQKE']
    square_size = 80   
    image_size = [square_size, square_size]
    init_weights_dir = 'data/model/Identify-Piece'
    init_weights_fname = 'cnn_pieces.h5' 
    epochs = 10
    tf_fit_kwargs = {'verbose':2}
    SQ_LABEL_COLUMNS = ['fname', 'SqColor-BWE', 'PcColor-BWE', 'PcType-PRNBQKE','HumCheck-YN'] 

    print('Initialize without file')
    piece_classifier = Piece_Classification(model_dir)
    print('Initialize with file')
#   piece_classifier = Piece_Classification(model_dir, model_weight_fname)
#   print(piece_classifier.model)

    #make X y data sample run
    piece_classifier.make_X_y(source_train, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path, image_size)

#   piece_classifier.train(source_train, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path, image_size, model_dir, model_weight_fname, init_weights_dir=init_weights_dir, init_weights_fname=init_weights_fname , epochs=epochs, test_data_dir=None)
#   piece_classifier.train(source_train, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path, image_size, model_dir, model_weight_fname, init_weights_dir=init_weights_dir, init_weights_fname=init_weights_fname , epochs=epochs, test_data_dir=source_valid)
    
#   piece_classifier.train(source_train, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path, image_size, model_dir, model_weight_fname, epochs=epochs, test_data_dir=None)
    piece_classifier.train(source_train, labels_csv_full_path, fname_col, label_cols, label_dict_csv_full_path, image_size, model_dir, model_weight_fname, epochs=epochs, test_data_dir=source_valid)

#   _ = piece_classifier.predict(source_test, image_size)
#   print(_)

#   piece_classifier.update_labels(main_source_dir, labels_csv_full_path, SQ_LABEL_COLUMNS, update_fn = gcb_utils.square_insert_default_values, update_fn_kwargs ={'label_cols':['SqColor-BWE', 'PcColor-BWE', 'PcType-PRNBQKE'], 'hum_check_col':['HumCheck-YN']})

