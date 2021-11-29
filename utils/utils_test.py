import utils
assert utils.make_dividable(0,100,8) == (2,98)

'''
utils.insert_sq_fnames('../data/raw/squares', '/Users/artun/Desktop/CodePlay/Get-Chess-Board/data/model/sq_labels.csv') 
'''

#Test image display size - conserving aspect ratio        
assert utils.get_new_im_size(1200, 800, 1000, 600) == (900, 600)
assert utils.get_new_im_size(1600, 800, 1000, 600)  == (1000, 500)
assert utils.get_new_im_size(1200, 600, 1000, 600) == (1000, 500)
assert utils.get_new_im_size(1000, 800, 1000, 600) == (750, 600)
assert utils.get_new_im_size(400, 400,  1000, 600) == (400, 400)




SQ_LABEL_COLUMNS = ['fname', 'SqColor-BW', 'PcColor-BWE', 'PcType-PRNBQK','HumCheck-YN'] 
SCREENSHOT_LABEL_COLUMNS = ['fname', 'height_pxl','width_pxl','label','x_min_pxl','y_min_pxl','x_max_pxl','y_max_pxl', 'HumCheck-YN']

def run_sq_label_update():
    utils.insert_data_fnames('../data/raw/squares', '/Users/artun/Desktop/CodePlay/Get-Chess-Board/data/model/sq_labels.csv', SQ_LABEL_COLUMNS)

    utils.update_sq_labels('../data/raw/squares', '/Users/artun/Desktop/CodePlay/Get-Chess-Board/data/model/sq_labels.csv') 


def run_screenshot_label_update():
    utils.insert_data_fnames('../data/raw/screenshots', '/Users/artun/Desktop/CodePlay/Get-Chess-Board/data/model/screenshot_boundboxes.csv', SCREENSHOT_LABEL_COLUMNS, update_fn=utils.screenshot_height_width_update)
    
    utils.update_screenshot_labels('../data/raw/screenshots', '/Users/artun/Desktop/CodePlay/Get-Chess-Board/data/model/screenshot_boundboxes.csv') 
run_sq_label_update()
run_screenshot_label_update()

