import csv
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
from PIL import Image, ImageTk
import random
import shutil
import tkinter


def divide_board(image, left_top, right_bottom, rows=8, cols=8):
    '''Divides the images into "rows" rows and "cols" columns starting from the "left_top" (x,y) pixel to the "right_bottom" [incl] (x,y) pixel. Returns the results in a list of tuples of form (row_no, col_no, square_image)'''


    squares = []
    left, top = left_top
    right, bottom = right_bottom
    
    top, bottom = make_dividable(top, bottom, rows) #what if not square? tbd 
    left, right = make_dividable(left, right, cols)
    print((left, top),(right, bottom))
    row_step = (bottom + 1 - top)//rows
    print('row_step:{}'.format(row_step))
    col_step = (right + 1 - left)//cols
    print('col_step:{}'.format(col_step))

    image = image[top:bottom+1,left:right+1,:]
#    plt.imshow(image)
#    plt.show()
    print(image.shape)
    image_divided = image.reshape(rows, row_step,cols, col_step, image.shape[-1]).swapaxes(1,2)
        
    for r in range(rows):
        for c in range(cols):
            squares.append((r,c,image_divided[r,c,:,:,:].reshape(row_step, col_step, image.shape[-1])))
    return squares
 

def make_dividable(low, high, groups):
    '''Helper function to create an interval of length "high"-"low" divisible by "groups". Squeezes the interval from both sides. Returns new (low, high)'''
    nudge_low = True
    while (high - low) % groups !=(groups-1):
        if nudge_low:
            low += 1
            nudge_low = False
        else:
            high -= 1
            nudge_low = True
    return (low, high)

def boards_to_squares(board_fnames, square_path):
    '''Processes the list "board_fnames" along with their "board_cols" and "board_rows" (int or list of ints) and writes as separate files to square_path. Output filenames are in the form: board_fname+'_squ_RxCy' where x, y are row and column numbers from the board.''' 
    for i, b_fn in enumerate(board_fnames):
        image = plt.imread(b_fn)
        squares = divide_board(image, (0,0), (image.shape[0], image.shape[1]))
        for s in squares:
            plt.imsave(square_path+'/'+b_fn.rsplit('/')[-1][:-4]+'_squ_R{0}_C{1}.png'.format(s[0], s[1]),s[2])

def convert_screenshot_to_square(
                screenshot_csv_full_path,
                screenshot_path,
                sq_csv_full_path,
                sq_files_path):
    '''Converts labeled screenshots to squares and transfers them to a directory for labeling. Expects input from the user whether to transfer squares from a certain screenshot.
        screenshot_csv_full_path. [Design choice for transferring only chess squares and not ads that can pop up at times.]
        Args:
            screenshot_csv_full_path: Full relative/absolute path to the screenshot bounding box csv file.
            screenshot_path: Relative/absolute path to the screenshot directory [source for files].
            sq_csv_full_path: Full rel/abs path to square labeling csv file. [to avoid duplicate transfers]
            sq_files_path: Destination path (rel/abs) for the square images.
        Returns:
            None. During execution displays the board to be imported and asks for a decision
        Notes: Non-chess-piece squares could also be imported for training purposes. Have to think/experiment.'''
  
    sq_df = pd.read_csv(sq_csv_full_path)
    sq_df.head()
    sq_existing_fnames = list(sq_df['fname'].map(lambda x: x.split('_')[0]).unique())

    scr_df = pd.read_csv(screenshot_csv_full_path)
    scr_df[['x_min_pxl', 'y_min_pxl', 'x_max_pxl', 'y_max_pxl']] = scr_df[['x_min_pxl', 'y_min_pxl', 'x_max_pxl', 'y_max_pxl']].astype('int') 


    screenshot_train_fnames = list(set([_[:-4] for _ in os.listdir(screenshot_path) if not _[0]=='.' and _[-4:]=='.png']))



    screenshots_to_board = list(set(screenshot_train_fnames) - set(sq_existing_fnames))
    print(len(screenshots_to_board))
    print('Files to be transferred: (Check them):\n',screenshots_to_board)


    
    for _screenshot in screenshots_to_board:
        if _screenshot == '':
            continue
        im_full_path = '/'.join([screenshot_path, _screenshot])+'.png'
        im = plt.imread(im_full_path)
        scr_mask = scr_df['fname'] == (_screenshot+'.png')
        im_data = scr_df[scr_mask]
        left_top = im_data[['x_min_pxl', 'y_min_pxl']].values[0]

        right_bottom = im_data[['x_max_pxl', 'y_max_pxl']].values[0]

        squares = gcb_utils.divide_board(im, left_top, right_bottom, rows=8, cols=8)
        rows, cols = 8, 8

        left, top = left_top
        right, bottom = right_bottom

        tmp_im =im[top:bottom+1,left:right+1,:]
        plt.imshow(tmp_im)
        plt.show()

        print('(p)rocess, (s)kip, (q)uit')
        inp = input()
        if inp == 'p':
            pass
        elif inp == 's':
            continue
        else:
            break

        for r in range(rows):
            for c in range(cols):
                _sq_im = squares[r*(rows)+c][2]
                _sq_name = _screenshot + '_brd_sqr_R{0}_C{1}.png'.format(str(r), str(c))
                plt.imsave('/'.join([sq_files_path,_sq_name]), _sq_im)




                 
SQ_LABEL_COLUMNS = ['fname', 'SqColor-BWE', 'PcColor-BWE', 'PcType-PRNBQKE','HumCheck-YN']
SCREENSHOT_LABEL_COLUMNS = ['fname', 'height_pxl','width_pxl','label','y_min_pxl','x_min_pxl','y_max_pxl','x_max_pxl']


def insert_data_fnames(data_path, labels_fname, label_file_columns, update_fn=None, update_fn_kwargs=None):
    '''Inserts the names of files not already in "data_path" into "labels_fname" csv file. Else, creates a labels file in csv format, using  "label_file_columns".. File contains column names in first row and data in the rest. This program inserts those names not already in "date__path" and applies update_fn (if provided) to the appended files. Prints the number of lines inserted. Returns None
    data_path: str, path of directory containing data files, relative to project root
    labels_fname: str, path of the csv file, relative to project root
    label_file_columns: list of str, ordered columns for the labels file
    update_fn: optional update function to manipulate the newly inserted data.
    update_fn_kwargs: dict, kwargs for update_fn, excludes df - the first arg of update_fn.'''
 
    if not os.path.exists(labels_fname):
        tmp_df = pd.DataFrame(columns=label_file_columns)
        tmp_df.to_csv(labels_fname, index=False)
        print('ATTENTION: New label file created with name:{}.\nCheck for any misspellings.'.format(labels_fname))
    images_in_file = []
    df = pd.read_csv(labels_fname)
    images_in_file = list(df['fname'].values)
    print(images_in_file)

    images_in_dir = [_ for _ in os.listdir(data_path) if _[-3:]=='png']

    images_to_append = sorted(list(set(images_in_dir)-set(images_in_file)))
   
    df_append = pd.DataFrame(columns=df.columns)
    df_append['fname'] = images_to_append
    
    if update_fn is not None:
        df_append = update_fn(df_append, **update_fn_kwargs)
 
    df_to_csv = pd.concat([df, df_append])
    df_to_csv.reset_index()
    df_to_csv.to_csv(labels_fname, mode='w', index=False, header=True)
    return None

def square_insert_default_values(df, label_cols=None, hum_check_col=None):
    '''Designed for use with insert_data_fnames for squares. Modifies the df to insert defaults for label_cols (default: E)  and hum_check_col (default N)
    Args:
        df: pd.DataFrame to be modified and later on appended.
        label_cols: list of str, columns of labels. Will be set to 'E' as default
        hum_check_cols: list of str, human check column. Will be set to 'N' as default.
    Returns:
        A dataframe with default values in indicated columns'''

    df[label_cols] = 'E'
    df[hum_check_col] = 'N'
    return df 

def screenshot_height_width_update(df, screenshot_path):
    '''Designed for use with insert_data_fnames for screenshots. Modifies the df to insert image height and width.
    Args:
        df: pd.DataFrame to be modified and later on appended.
        screenshot_path: str, relative/absolute path to screenshot images
    Returns:
        A dataframe with new image heigh and width columns'''

    for _i, _r in df.iterrows():
        im = Image.open(screenshot_path+'/'+_r['fname'])
        df.loc[_i, 'height_pxl'] = int(im.height)
        df.loc[_i, 'width_pxl'] = int(im.width)
    return df


def update_sq_labels(square_path, sq_labels_fname):
    '''Updates square labels
    square_path: Relative path to the chess square image directory
    sq_labels_fname: Relative path of the square labels file
    '''

    if not os.path.exists(sq_labels_fname):
        print('File Name \'{0}\' does not exist.'.format(sq_labels_fname))
         
    if not os.path.exists(square_path):
        print('Path \'{0}\' does not exist.'.format(square_path))

    #pdb.set_trace()
    labels_df = pd.read_csv(sq_labels_fname)
    #print(labels_df.columns)
    #print(labels_df.head())
    
    root = tkinter.Tk()
    square_colors = ('B', 'W')
    piece_colors = ('B', 'W', 'E')
    piece_types = ('P', 'R', 'N', 'B', 'Q', 'K', 'E')
    human_check = ('Y', 'N')
    label_ct = 16
    row_ct = 0
 
    labels = []
    square_color_radios = []
    piece_color_radios = []
    piece_type_radios = []
    human_check_radios = [] 
    
    #insert headers and adjust columns
    col_img_bgn = 0
    tkinter.Label(root, text='Image').grid(row=row_ct, column=col_img_bgn)
    col_sq_bgn = col_img_bgn + 1
    tkinter.Label(root, text='Square Color: (B)-(W)').grid(row=0, column=col_sq_bgn, columnspan= len(square_colors))
    col_clr_bgn = col_sq_bgn + len(square_colors)
    tkinter.Label(root, text='Piece Color: (B)-(W)-(E)').grid(row=0, column=col_clr_bgn, columnspan= len(piece_colors))
    col_typ_bgn = col_clr_bgn + len(piece_colors)   
    tkinter.Label(root, text='Piece Type: (P)-(R)-(N)-(B)-(Q)-(K)-(E)').grid(row=0, column=col_typ_bgn, columnspan=len(piece_types))
    col_hum_bgn = col_typ_bgn+ len(piece_types)
    tkinter.Label(root, text='Human Check: (Y)-(N)').grid(row=0, column=col_hum_bgn, columnspan=len(human_check))
    
    crnt_label_idx = 0
    last_label_idx = min(label_ct, len(labels_df)-crnt_label_idx)
    crnt_label_df = labels_df.loc[range(crnt_label_idx,last_label_idx),:]
    crnt_label_df_index = crnt_label_df.index.values
    for i in range(label_ct):
        square_color_radios.append([tkinter.StringVar()]) 
        piece_color_radios.append([tkinter.StringVar()]) 
        piece_type_radios.append([tkinter.StringVar()])
        human_check_radios.append([tkinter.StringVar()])


    #start inserting labeling rows

    def clear_rows():
        for i in range(label_ct):
            labels[i].destroy()
            _cols = [square_color_radios, piece_color_radios, piece_type_radios, human_check_radios]
            for _c in _cols:
                while len(_c[i])>1:
                    _c[i].pop().destroy()

    def insert_radios(var_list, button_list, button_row_index, frame_col_bgn, csv_col_name):
        '''Insert series of connected radio butttons in given rows, starting at indicated columns
    var_list: list of radio button choices. e.g. ['B','W']
    button_list: mixed list of radio button objects. First element is tkinter.StringVar, the rest (len(var_list)) are tkinter.Radiobuttons connected to the tkinter.StringVar.
    button_row_index: index of the row in the data portion of the app
    frame_col_bgn: Starting column of the buttons in the frame'''
 
        set_data_idx = None
        button_list[button_row_index][0].set('')
        for j, s in enumerate(var_list):
            button_list[button_row_index].append(tkinter.Radiobutton(root, text=s, variable=button_list[button_row_index][0], value=s))
            button_list[button_row_index][j+1].grid(row=button_row_index+1, column = frame_col_bgn+j)
            #pdb.set_trace()
            if crnt_label_df.iloc[button_row_index][csv_col_name] == s:
                set_data_idx = j+1
        if set_data_idx:
            button_list[button_row_index][set_data_idx].invoke()
            set_data_idx = None 

    def on_save_frame_and_next():
        """Write all data in current frame to the csv file and display the next frame"""
        nonlocal last_label_idx
        nonlocal crnt_label_df
        for i in range(label_ct): #Update labels directly, not via crnt_label
            labels_df.loc[(crnt_label_df.index)[i],'SqColor-BWE'] = square_color_radios[i][0].get()
            labels_df.loc[(crnt_label_df.index)[i],'PcColor-BWE'] = piece_color_radios[i][0].get()
            labels_df.loc[(crnt_label_df.index)[i],'PcType-PRNBQKE'] = piece_type_radios[i][0].get()
            labels_df.loc[(crnt_label_df.index)[i],'HumCheck-YN'] = human_check_radios[i][0].get()
        crnt_label_idx = last_label_idx
        last_label_idx = crnt_label_idx + min(label_ct, len(labels_df)-crnt_label_idx)
        #print('crnt_label_idx: {0}\nlast_label_idx: {1}'.format(crnt_label_idx,last_label_idx))
        crnt_label_df = labels_df.iloc[crnt_label_idx:last_label_idx,:]
        #print(type(crnt_label_df)) 
        #print(crnt_label_df.shape) 
        #pdb.set_trace()
        labels_df.to_csv(sq_labels_fname, mode='w', index=False, header=True)
        clear_rows()     
        draw_rows()
    def on_mark_checked():
        """Mark all Human Checked on the frame as 'Y'"""
        for i in range(label_ct): #Update labels directly, not via crnt_label
            human_check_radios[i][0].set('Y')
    
    def on_make_estimates():
        """Estimate piece color and type using the Square Labeler Network  and set human check to 'N'""" 
        for i in range(label_ct): #Update labels directly, not via crnt_label
            human_check_radios[i][0].set('N')

    def draw_rows():
        nonlocal labels 
        labels = []
        set_data_idx = None
        for i in range(label_ct):
            #insert image
            im = Image.open(square_path+'/'+crnt_label_df.iloc[i,0])
            im = im.resize((30,30))
            ph = ImageTk.PhotoImage(im)
            labels.append(tkinter.Label(root, image=ph))
            labels[i].image = ph 
            labels[i].grid(row=i+1, column = col_img_bgn)

            insert_radios(square_colors, square_color_radios, i, col_sq_bgn,'SqColor-BWE')
            insert_radios(piece_colors, piece_color_radios, i, col_clr_bgn, 'PcColor-BWE') 
            insert_radios(piece_types, piece_type_radios, i, col_typ_bgn, 'PcType-PRNBQKE')
            insert_radios(human_check, human_check_radios, i, col_hum_bgn, 'HumCheck-YN') 

    draw_rows()     
    make_estimates_button = tkinter.Button(root, text="Make Estimates", command = on_make_estimates)
    make_estimates_button.grid(row=label_ct+1, column = 1, columnspan=2)
    mark_checked_button = tkinter.Button(root, text="Mark As Checked", command = on_mark_checked)
    mark_checked_button.grid(row=label_ct+1, column = 3, columnspan=3)
    next_button = tkinter.Button(root, text="Save Frame & Next", command = on_save_frame_and_next)
    next_button.grid(row=label_ct+1, column = 13, columnspan=2)
    root.mainloop()    
    #print('end')

def get_new_im_size(w, h, w_max, h_max):
    '''Resize image conserving aspect ratio with width and height lower than specicied max pixel values.
    w: width of the image in pixels
    h: height of the image in pixels
    w_max: maximum width of the image in pixels
    h_max: maximum height of the image in pixels'''

    if w==0 or h==0 or w_max==0 or h_max==0:
        raise Exception('Parameters cannot be <=0. Should be >0.')
    
    if h < h_max and w < w_max:
       return int(w), int(h)
    
    w_to_h = w/h
    wmax_to_hmax = w_max/h_max

    if w_to_h >= wmax_to_hmax:
        return int(w_max), int(h*w_max/w)
    else:
        return int(w*h_max/h), int(h_max) 

def get_im_shrink(w, h, w_max, h_max):
    '''Calculate shrink factors from original image to display
    w: width of the image in pixels
    h: height of the image in pixels
    w_max: maximum width of the image in pixels
    h_max: maximum height of the image in pixels'''
    im_w_new, im_h_new = get_new_im_size(w, h, w_max, h_max)
    im_w_shrink, im_h_shrink,  =im_w_new/w, im_h_new/h 
    return im_w_shrink, im_h_shrink 

def scale_label_coords(df_file, img_width_col, img_height_col, disp_max_width, disp_max_height, label_cols, is_file_to_disp=True):
    '''Returns a new dataframe with label coordinates appropriate for image display settings
    df: dataFrame from the csv file  
    img_width: name of file image width column
    img_height: name of file image height column
    disp_max_width: maximum width of the image in pixels
    disp_max_height: maximum height of the image in pixels 
    label_cols: list - names of x and y coordinates of label columns
    returns dataframe with label coordinates adjusted for display size.
    '''
    
    df_img = df_file.copy()
    shrink_inp = [list(df_img[img_width_col]), list(df_img[img_height_col]), [disp_max_width]*len(df_img), [disp_max_height]*len(df_img)] 
    shrink_fac = [get_im_shrink(*_)[0] for _ in zip(*shrink_inp)] #note to self: Maybe vectorize.
    df_img['shrink_fac'] = shrink_fac
    if not is_file_to_disp:
        df_img['shrink_fac'] = 1/df_img['shrink_fac']
    for _ in label_cols:
        df_img[_] = np.round(df_img[_]*df_img['shrink_fac']) #at times f(finv)<>f due to rounding.
    df_img.drop(['shrink_fac'], axis=1, inplace=True)
    return df_img


def update_screenshot_labels(screenshot_path, screenshot_labels_fname):
    '''GUI to update screenshot labels and boundingboxes
    screenshot_path: Relative path to the chess screenshot image directory
    screenshot_labels_fname: Relative path of the screenshot labels file
    '''

    if not os.path.exists(screenshot_labels_fname):
        print('File Name \'{0}\' does not exist.'.format(screenshot_labels_fname))
         
    if not os.path.exists(screenshot_path):
        print('Path \'{0}\' does not exist.'.format(screenshot_path))
    
    MAX_DISP_HEIGHT = 600 #1800
    MAX_DISP_WIDTH = 1000 #2880

    file_labels_df = pd.read_csv(screenshot_labels_fname)
    labels_df =  scale_label_coords(file_labels_df, 'width_pxl', 'height_pxl', MAX_DISP_WIDTH, MAX_DISP_HEIGHT, ['x_min_pxl', 'x_max_pxl', 'y_min_pxl', 'y_max_pxl'], is_file_to_disp=True)

    root = tkinter.Tk()
    root.title('Update Screenshot Labels [Click-Drag-Release to Add Label]')

    scr_fnames = list(set(labels_df['fname']))
    if len(scr_fnames)>0:
        scr_i = 0
   
    crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]

    canvas = tkinter.Canvas(root)
   
    row_ct = 0
    labels= []
    im_h_shrink, im_w_shrink = 1, 1
    def save_pos(event):
        global lastx, lasty
        lastx, lasty = event.x, event.y

    def addRectangle(event):
        nonlocal canvas, crnt_scr_gr
        _cols = ['label','x_min_pxl','y_min_pxl','x_max_pxl','y_max_pxl','HumCheck-YN']
        empty_processed = False
        for _i, _r in crnt_scr_gr.iterrows():
            if any(pd.isna(labels_df.loc[_i,_cols])) and not empty_processed:
                #print('got empty')
                labels_df.loc[_i, _cols] = ['Chessboard', min(lastx, event.x), min(lasty, event.y), max(lastx, event.x), max(lasty, event.y), 'Y']
                crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]
                empty_processed = True
        if empty_processed == False:
            #pdb.set_trace()
            new_idx = labels_df.index.max()+1
            labels_df.loc[new_idx,:] = crnt_scr_gr.iloc[0,:]
            labels_df.loc[new_idx, _cols] = ['Chessboard', min(lastx, event.x), min(lasty, event.y), max(lastx, event.x), max(lasty, event.y), 'Y']
        crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]
        draw_canvas()
        draw_labels()
        draw_rows()
                
        save_pos(event)
    
    def draw_canvas():
        nonlocal canvas, im_h_shrink, im_w_shrink
        #insert image with smaller size, same aspect ratio
        im = Image.open(screenshot_path+'/'+scr_fnames[scr_i])
        im_h_orig, im_w_orig = im.height, im.width
        im_h_new, im_w_new = get_new_im_size(im.height, im.width, MAX_DISP_HEIGHT, MAX_DISP_WIDTH)
        im_h_shrink, im_w_shrink,  =im_h_new/im_h_orig, im_w_new/im_w_orig 
        im = im.resize((im_w_new, im_h_new))
        ph = ImageTk.PhotoImage(im)

        #pdb.set_trace()
        canvas.configure(width=im_w_new, height=im_h_new)
        canvas.test = ph
        canvas.create_image(0,0,image=ph, anchor="nw")
        canvas.grid(row=0, column=0, columnspan=len(labels_df.columns)+4)
        canvas.bind("<Button-1>", save_pos)
        canvas.bind("<B1-ButtonRelease>", addRectangle)

    def draw_labels():
        _label_cols = ['label','x_min_pxl','y_min_pxl','x_max_pxl','y_max_pxl'] 
        for _i, _r in crnt_scr_gr.iterrows():
            #pdb.set_trace()
            if not any(pd.isna(labels_df.loc[_i,_label_cols])):
                _v = tuple(labels_df.loc[_i,_label_cols].values)
                canvas.create_rectangle(_v[1:], outline='white')
                canvas.create_text(_v[1:3], text=_v[0], fill='white', anchor="nw")
        
    #Insert headers 
    row_ct += 1
    header_row = row_ct
    labels = []
    
    def draw_header():
        '''Draws header from the columns of the dataset on the header row'''
        for _i, _c in enumerate(labels_df.columns):
            _h = tkinter.Label(text=_c, font=('bold'))
            _h.grid(row=header_row, column=_i)

    def delete_row(i):
        nonlocal crnt_scr_gr
        #print("Row: {} deleted.".format(i))
        _cols = ['label','x_min_pxl','y_min_pxl','x_max_pxl','y_max_pxl','HumCheck-YN']
        if len(crnt_scr_gr)==1:
           for _c in _cols:
               labels_df.loc[i,_c] = np.nan 
        else:
           labels_df.drop(index=i, inplace=True)
        crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]
        draw_rows()
    

    def draw_rows():
        nonlocal row_ct, labels, next_button
        '''Draws rows from the current screenshot data'''
        for _ in labels:
            #pdb.set_trace()
            _.destroy()
        labels = []
        
        #Insert label rows
        for _i, _r in crnt_scr_gr.iterrows():
            row_ct += 1
            for _j, _d in enumerate(_r):
                labels.append(tkinter.Label(text=_d))
                labels[-1].grid(row=row_ct, column=_j)
            _button = tkinter.Button(root, text='Delete', command=partial(delete_row, _r.name))
            labels.append(_button)
            _button.grid(row=row_ct, column=_j+2)

        draw_canvas()
        draw_labels() 
        next_button.grid(row=row_ct+2, column = 10, columnspan=1)


    def on_save_frame_and_next(save_frame=True):
        nonlocal scr_i, crnt_scr_gr, next_button

        #Save labels to file (after appropriate scaling)        
        if save_frame:
            file_labels_df =  scale_label_coords(labels_df, 'width_pxl', 'height_pxl', MAX_DISP_WIDTH, MAX_DISP_HEIGHT, ['x_min_pxl', 'x_max_pxl', 'y_min_pxl', 'y_max_pxl'], is_file_to_disp=False)
            file_labels_df.to_csv(screenshot_labels_fname, index=False)
        else:
            print('not saved')
    
        
        scr_fnames = list(set(labels_df['fname']))
        scr_i = (scr_i+1)%len(scr_fnames) #Moves to the next image
        next_scr_i = scr_i+2 if scr_i+2<=len(scr_fnames) else 1 #for displaying the one after
        next_button.configure(text="Save Frame & Next to {0}/{1}".format(next_scr_i, len(scr_fnames)))
        crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]

        draw_canvas()
        draw_labels()
        draw_header() 
        draw_rows()


    draw_canvas()
    draw_labels()
    draw_header() 
    next_button = tkinter.Button(root, text="Save Frame & Next to {0}/{1}".format(scr_i+2, len(scr_fnames)), command = on_save_frame_and_next)
    draw_rows()
                
    next_button.grid(row=row_ct+2, column = 10, columnspan=1)
    root.bind("<Right>", func=lambda x: on_save_frame_and_next(save_frame=False))
    root.mainloop()


def prepare_scr_input_for_yolov5(source_csv_path, fname_col, label_col, size_cols,  min_corner_cols, max_corner_cols, image_path, yaml_file_path, train_path, validation_path):
    '''Prepare input (yaml file and individual .txt files) to be used for yolov5.

    Uses the csv file to prepare a .yaml file and the text data files for each image. [to be separated later into two functions: yaml & text data]
    Args:
        source_csv_path: path for the csv file containing bounding boxes.
        fname_col: column name for the fname 
        label_col: column name for the label 
        size_cols: column names of (width, height) pairs.
        min_corner_cols: column names of (x_min, y_min) of bounding boxes.
        max_corner_cols: column names of (x_max, y_max) of bounding boxes.
        image_path: path to the image directory
        yaml_file_path: path to the yaml file
        train_path: path to the training directory [only used for yaml file at this time]
        validation_path:  path to the training directory [only used for yaml file at this time]

    Returns: 
        None. Prints success status.
    '''
    full_labels_df = pd.read_csv(source_csv_path)
   
    #filter for valid labels [note: missing Human Check]
    _cols = [label_col]
    _cols.extend(min_corner_cols)
    _cols.extend(max_corner_cols)
    labels_df = full_labels_df[~pd.isna(full_labels_df[_cols]).any(axis=1)]
 
    #write yaml file
    label_uniq_set = list(set(labels_df[label_col]))
    label_uniq_set.sort()
    nc = len(label_uniq_set)
    class_dict = dict(zip(label_uniq_set, range(nc)))  #this dict should really be input.
    #print(label_uniq_set)
    #print(nc)
    #print(class_dict)
 
    yaml_str = ''
    yaml_str += 'train: '+ train_path+'\n'
    yaml_str += 'val: '+ validation_path+'\n'
    yaml_str += '\n'
    yaml_str += 'nc: {0:d}\n'.format(nc)
    yaml_str += 'names: {0}'.format(label_uniq_set)
   
    with open(yaml_file_path, 'w') as f:
        f.write(yaml_str)
 
    print('yaml file written as follows:\n{0}\n'.format(yaml_str))

   #write text data
    x_min, x_max = min_corner_cols[0], max_corner_cols[0]
    y_min, y_max = min_corner_cols[1], max_corner_cols[1]
    width = size_cols[0]
    height = size_cols[1]
 
    
    labels_df['class_id'] = labels_df[label_col].map(class_dict)

    labels_df['rel_center_x'] = (labels_df[x_max] + labels_df[x_min])/2/labels_df[width]
    labels_df['rel_center_y'] = (labels_df[y_max] + labels_df[y_min])/2/labels_df[height]
    
    labels_df['rel_width'] = (labels_df[x_max] - labels_df[x_min])/labels_df[width]
    labels_df['rel_height'] = (labels_df[y_max] - labels_df[y_min])/labels_df[height]


    for _fn, _g in labels_df.groupby(by=fname_col):
         with open(image_path+'/'+_fn[:-4]+'.txt', 'w') as f:
             for _i, _r in _g.iterrows(): #following yolov5 annotation [id, [0,1]^4]
                 f.write('{0:d} '.format(_r['class_id']))
                 f.write('{0:f} '.format(_r['rel_center_x']))
                 f.write('{0:f} '.format(_r['rel_center_y']))
                 f.write('{0:f} '.format(_r['rel_width']))
                 f.write('{0:f}\n'.format(_r['rel_height']))
     
    print('Data files written.')
    
def split_train_valid_test(source_csv_path, fname_col, image_path, train_path, validation_path, test_path, train_ratio, validation_ratio, img_size, img_format='jpg', random_seed=None):
    '''Splits images into train, validation and test sets. Transfers them to relevant directories with indicated size and format.
 
       Sum of train and validation ratio cannot be >=1. Rest allocated to test.
    Args:
        source_csv_path: path for the csv file containing bounding boxes.
        fname_col: column name for the fname  
        image_path: path to the image directory
        train_path: path to the training directory 
        validation_path:  path to the validation directory
        test_path:  path to test directory
        train_ratio: ratio of training images to all images in source_csv_path
        validation_ratio: ratio of validation images to all images in source_csv_path
        img_size: (w,h) size of the train, validation and test images in pixels
        img_format: format of train, validation and test images [default:'jpg']
        shuffle_seed: int, seed to shuffle list - works across runs if sorted file list is the same 
    Returns:
        None. Prints out the counts of final allocation.
    '''        
    if not ((0<train_ratio<=1) and (0<=validation_ratio<1) and (0<train_ratio+validation_ratio<=1)):
        raise Exception('Ratio error: Train -> (0,1], Validation -> [0,1), their sum -> (0,1]\n Your inputs are: \nTrain:{0:f}\nValidate:{1:f}'.format(train_ratio, validation_ratio))

    IMG_FORMATS = ['png', 'jpg', 'jpeg']

    #Get labeled source files names
    df_source = pd.read_csv(source_csv_path)
    notna_cols = df_source.columns  #here as a reminder to push notna_cols to Args if source csv format changes
    not_na_labels_df = df_source[~pd.isna(df_source[notna_cols]).any(axis=1)]
    #get is na from source files

    #Get file names in image_path
    images_in_dir = [_ for _ in os.listdir(image_path) if _[-3:]=='png']
    
    images_to_split = set(not_na_labels_df[fname_col]).intersection(set(images_in_dir))
    images_to_split = list(images_to_split)
    images_to_split = sorted(images_to_split)

    if random_seed:
        random.shuffle(images_to_split)
    else:
        random.Random(random_seed).shuffle(images_to_split)

    len_imgs = len(images_to_split)
    train_bgn = 0
    validation_bgn = int(train_ratio*len_imgs)
    test_bgn = int((train_ratio + validation_ratio)*len_imgs)
    if validation_bgn == train_bgn:
       validation_bgn += 1
    if test_bgn == validation_bgn:
       test_bgn = min(len_ims, test_bgn+1)

    train_imgs = images_to_split[train_bgn:validation_bgn]
    validation_imgs = images_to_split[validation_bgn:test_bgn]
    test_imgs = images_to_split[test_bgn:len_imgs]
    
    print("Total Train Images: {}".format(len(train_imgs)))
    print("Total Validation Images: {}".format(len(validation_imgs)))
    print("Total Test Images: {}".format(len(test_imgs)))
     
    dests = [['Train', train_path, train_imgs] , ['Validation', validation_path, validation_imgs], ['Test', test_path, test_imgs]]   
 
    print('Files in following directories will be deleted and repopulated:\n{0}'.format([_[1] for _ in dests]))
    print("Do you agree? (Enter y or Y for Yes, any other string is a No)")
    response = input()
    if (response == 'y') or (response == 'Y'):
        print("User accepted: Repopulating directories")
    else:
        raise Exception('User declined to delete/overwrite directory. Exiting...')
    for d in dests:
        _stage, _path, _imgs = d
        
        #delete files
        if os.path.exists(_path):
            for _f in os.listdir(_path):
               _ext = _f.split('.')[-1]
               if _ext in IMG_FORMATS or _ext == 'txt':
                   os.remove(_path+'/'+_f)
        else:
           os.mkdir(_path)
      
        for _f in _imgs:
            #Resize, convert, write image in _dest dir
            _im = Image.open(image_path+'/'+_f)
            _im = _im.resize(img_size)
            _im = _im.convert('RGB')
            _ = _f.split('.')
            _[-1] = img_format
            _fi = '.'.join(_)
            _im.save(_path+'/'+_fi)

            #Insert labeling text
            _ = _f.split('.')
            _[-1] = 'txt'
            _ft = '.'.join(_)
            shutil.copy(image_path+'/'+_ft, _path+'/'+_ft)
            
           #there are still issues with how to treat the csv display on labeling. Not now, but you have more time to think until the next data ingest/labeling cycle.
 
