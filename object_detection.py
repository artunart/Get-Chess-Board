import gcb_utils.gcb_utils as gcb_utils
import os
import pandas as pd 
import torch
import yolov5
class Board_Detection():
    '''
    A wrapper around PyTorch implementation of Yolov5 (https://github.com/ultralytics/yolov5, see https://pjreddie.com/media/files/papers/yolo.pdf for Yolo) adapted for training and modeling chessboard captures from screenshots.
    '''
    
    def __init__(self, model_dir, model_weight_file=None):
        self.model_dir = model_dir
        self.model_weight_file = model_weight_file
        if model_weight_file is None:
            print('No model weight file specified. You can train a new model using Object_Detection.train.')
#        else: 
#            self.model = torch.hub.load(self.model_dir, self.model_weight_file, source = 'local')
        self.SCREENSHOT_LABEL_COLUMNS = ['fname', 'height_pxl','width_pxl','label','x_min_pxl','y_min_pxl','x_max_pxl','y_max_pxl', 'HumCheck-YN']

#    def _test_yolov5_detect(self, imgsz=(1440, 1440),**yolov5_kwargs):
#        '''Runs detect.py from yolov5 with yolov5 keyword args. (for debugging, if necessary)'''
#        yolov5.detect.run(imgsz, **yolov5_kwargs)



    def predict(self, model_dir, model_weight_fname, source_dir, imgsz, project_dir='tmp', project_name='tmpexp', exist_ok=True, **yolov5_kwargs):
        '''
        Outputs board predictions of files in source_dir with bounding boxes using the yolov5.detect function.
        Args:
             model_dir: path to the weights. Directory has to include hubconf.py
             model_weight_fname: filename for the model weights file ( .pt)
             source_dir: path for the images for which detection is to be performed
             imgsz: (w,h) tuple for image size in pixels. (width, height) Choose square, at least as large as input image.
             project_dir: path for the output. 'tmp' is default.
             project_name: project name for output. Directory stump to be enumerated under project_dir if exist_ok is False. 'tmpexp' is default.
             exist_ok: Will overwrite existing project dir. Will not increment. Be careful when tthis is set to False. Detections will take up space. Consider removing the contents of project dir from time to time.
        
        Returns:
            pd.DataFrame object consisting of fname, label, bounding box coordinates (x_min_norm, y_min_norm, x_max_norm, y_max_norm) and confidence columns.
        Notes: see yolov5.detect.run for a more complete list of argument for keywords.
       '''
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)
            print('Directory Created: {}'.format(os.path.abspath(project_dir)))
        #Directories under project_dir before detection [being careful about what to delete later on]
        pre_detection_dirs = [_ for _ in os.listdir(project_dir) if os.path.isdir('/'.join([project_dir,_]))] 
        

        #Run detection
        yolov5.detect.run(
                          weights = '/'.join([model_dir, model_weight_fname]),
                          source = source_dir,
                          imgsz = imgsz, 
                          project = project_dir,
                          name = project_name,
                          exist_ok = exist_ok,
                          **yolov5_kwargs)
        
        #Find newly added directory, if any.
        post_detection_dirs = [_ for _ in os.listdir(project_dir) if os.path.isdir('/'.join([project_dir,_]))]
        added_dir = list(set(post_detection_dirs).difference(set(pre_detection_dirs)))
        
        
        #Note current dir
        if exist_ok:
            crnt_project_dir ='/'.join([project_dir, project_name])
        else:
            crnt_project_dir = '/'.join([project_dir, added_dir[0]])
        
        #Note output dirs and other reminders
        print('yolov5 detection results are saved in: {}'.format(os.path.abspath(crnt_project_dir)))
        if not exist_ok: 
            print('The above directory is newly created, consider deleting new directories to save space')

        #Output detection results in a pd.DataFrame
        labels_dir = '/'.join([crnt_project_dir, 'labels'])
        labels_files =  [_ for _ in os.listdir(labels_dir) if _.endswith('.txt')]
        
        df_cols = ['fname', 'label', 'x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm', 'confidence']
        labels_df = pd.DataFrame(columns = df_cols)
        
        for _ in labels_files:
            _df = pd.read_csv('/'.join([labels_dir, _]), sep=' ')
            _df = pd.concat([pd.Series(data=[_[:-4]]*len(_df)), _df], axis=1)
            _df.columns = df_cols
            labels_df = pd.concat([labels_df, _df])

        return labels_df


    def train(self, yaml_path_or_dict, init_weights_dir, init_weights_fname, train_project_dir, train_project_name, imgsz = 1440, epochs=1500, cache = True, device = 'cpu',  **yolov5_kwargs):
        '''
        Trains a new model on a dataset using select. Note that all paths of directories need to be specified relative to the yolov5 directory
        '''
        
        #yolov5.train.ROOT = '.' #yolov5.train code has no option to set root - setting root this way
        p = yolov5.train.parse_opt()
        yolov5_kwargs['data'] = yaml_path_or_dict
        yolov5_kwargs['weights'] = '/'.join([init_weights_dir, init_weights_fname])
        yolov5_kwargs['project'] = train_project_dir
        yolov5_kwargs['name'] = train_project_name
        yolov5_kwargs['imgsz'] = imgsz
        yolov5_kwargs['epochs'] = epochs
        yolov5_kwargs['cache'] = cache
        yolov5_kwargs['device'] = device
        for _k, _v in yolov5_kwargs.items():
            setattr(p, _k, _v) 
        print(p)      
        #yolov5_kwargs[
        yolov5.train.main(p)



#maybe needs list of files here, too, to replicate a refence run.
    
    def update_labels(self, 
                      data_path, 
                      labels_fname, 
                      screenshot_label_columns=None, 
                      update_fn = gcb_utils.screenshot_height_width_update):
       #Passage to label updating GUI.
       if screenshot_label_columns is None:
           screenshot_label_columns  = self.SCREENSHOT_LABEL_COLUMNS
       print('Data path: {}'.format(data_path))
       print('Labels path + filename: {}'.format(labels_fname))
       print('Screenshot label columns: {}'.format(screenshot_label_columns))
       print('Update function: {}'.format(update_fn))
       
       gcb_utils.insert_data_fnames(data_path, labels_fname, screenshot_label_columns, update_fn)
       gcb_utils.update_screenshot_labels(data_path, labels_fname)
    

if __name__ == '__main__':
    model_dir = 'yolov5/runs/train/exp21/weights'
    model_weight_fname = 'best.pt' 
    source_dir = 'data/raw/screenshots/test'
    imgsz = (1440, 1440)
    project_dir = 'data/model/detect'
    project_name = 'dev'
    conf_thres = 0.10
    
#    print(os.path.abspath('.'))
#    print(os.listdir(model_dir))
    board_detector = Board_Detection(model_dir, model_weight_fname)
#    prediction = board_detector.predict(model_dir, model_weight_fname, source_dir, imgsz, project_dir, project_name,  conf_thres, save_txt = True, save_conf = True)
#    prediction = board_detector.predict(model_dir, model_weight_fname, source_dir, imgsz, save_txt = True, save_conf = True)
#    print(prediction)
    yaml_path_or_dict = 'data/model/scr_data_cli.yaml'
    init_weights_dir = 'yolov5'
    init_weights_fname = 'yolov5s.pt' 
    train_project_dir = 'data/model/train'
    train_project_name = 'exp'
    epochs = 1
#    board_detector.train(yaml_path_or_dict, init_weights_dir, init_weights_fname, train_project_dir, train_project_name, epochs=epochs)

    screenshot_data_path = 'data/raw/screenshots'
    screenshot_labels_fname = 'data/model/screenshot_boundboxes.csv'
    board_detector.update_labels(screenshot_data_path, screenshot_labels_fname)





