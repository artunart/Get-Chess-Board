import pandas as pd
import utils
assert utils.make_dividable(0,100,8) == (2,98)

'''
utils.insert_sq_fnames('../data/raw/squares', '/Users/artun/Desktop/Springboard/springboard/Get-Chess-Board/data/model/sq_labels.csv') 
'''

#Test image display size - conserving aspect ratio        
assert utils.get_new_im_size(1200, 800, 1000, 600) == (900, 600)
assert utils.get_new_im_size(1600, 800, 1000, 600)  == (1000, 500)
assert utils.get_new_im_size(1200, 600, 1000, 600) == (1000, 500)
assert utils.get_new_im_size(1000, 800, 1000, 600) == (750, 600)
assert utils.get_new_im_size(400, 400,  1000, 600) == (400, 400)

df_file = pd.read_csv('/Users/artun/Desktop/Springboard/springboard/Get-Chess-Board/data/model/screenshot_boundboxes.csv')
df_disp = utils.scale_label_coords(df_file, 'width_pxl', 'height_pxl', 1000, 600, ['x_min_pxl', 'x_max_pxl', 'y_min_pxl', 'y_max_pxl'], is_file_to_disp=True)
utils.scale_label_coords(df_disp, 'width_pxl', 'height_pxl', 1000, 600, ['x_min_pxl', 'x_max_pxl', 'y_min_pxl', 'y_max_pxl'], is_file_to_disp=False)


SQ_LABEL_COLUMNS = ['fname', 'SqColor-BW', 'PcColor-BWE', 'PcType-PRNBQK','HumCheck-YN'] 
SCREENSHOT_LABEL_COLUMNS = ['fname', 'height_pxl','width_pxl','label','x_min_pxl','y_min_pxl','x_max_pxl','y_max_pxl', 'HumCheck-YN']

def run_sq_label_update():
    utils.insert_data_fnames('../data/raw/squares', '../data/model/sq_labels.csv', SQ_LABEL_COLUMNS)

    utils.update_sq_labels('../data/raw/squares', '../data/model/sq_labels.csv') 


def run_screenshot_label_update():
    utils.insert_data_fnames('../data/raw/screenshots', '../data/model/screenshot_boundboxes.csv', SCREENSHOT_LABEL_COLUMNS, update_fn=utils.screenshot_height_width_update)
    
    utils.update_screenshot_labels('../data/raw/screenshots', '../data/model/screenshot_boundboxes.csv') 
run_sq_label_update()
run_screenshot_label_update()

#utils.prepare_scr_input_for_yolov5('../data/model/screenshot_boundboxes.csv', 'fname', 'label', ['width_pxl' ,'height_pxl'], ['x_min_pxl', 'y_min_pxl'], ['x_max_pxl', 'y_max_pxl'], '../data/raw/screenshots', '../data/model/scr_data.yaml', '../data/raw/screenshots/train', '../data/raw/screenshots/validation')
 
#utils.split_train_valid_test('../data/model/screenshot_boundboxes.csv', 'fname', '../data/raw/screenshots',  '../data/raw/screenshots/train', '../data/raw/screenshots/validation', '../data/raw/screenshots/test', 0.7, 0.2, (1440,900))