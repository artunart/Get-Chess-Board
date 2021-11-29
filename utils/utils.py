import csv
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
from PIL import Image, ImageTk
import tkinter
from tkinter import ttk


def divide_board(image, top_left, bottom_right, rows=8, cols=8):
    '''Divides the images into "rows" rows and "cols" columns starting from the "top_left" pixel to the "bottom_right" [incl] pixel. Returns the results in a list of tuples of form (row_no, col_no, square_image)'''


    squares = []
    top, left = top_left
    bottom, right = bottom_right
    
    top, bottom = make_dividable(top, bottom, rows) #what if not square? tbd 
    left, right = make_dividable(left, right, cols)
    row_step = (bottom-top)//rows
    col_step = (right-left)//cols

    image = image[top:bottom,left:right,:]
    print(image.shape)
    image_divided = image.reshape(rows, row_step, cols, col_step, image.shape[-1]).swapaxes(1,2)
        
    for r in range(rows):
        for c in range(cols):
            squares.append((r,c,image_divided[r,c,:,:,:].reshape(row_step, col_step, image.shape[-1])))
    return squares
 

def make_dividable(low, high, groups):
    '''Helper function to create an interval of length "high"-"low" divisible by "groups". Squeezes the interval from both sides. Returns new (low, high)'''
    nudge_low = True
    while (high - low) % groups !=0:
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
                 
SQ_LABEL_COLUMNS = ['fname', 'SqColor-BW', 'PcColor-BWE', 'PcType-PRNBQK','HumCheck-YN']
SCREENSHOT_LABEL_COLUMNS = ['fname', 'height_pxl','width_pxl','label','y_min_pxl','x_min_pxl','y_max_pxl','x_max_pxl']


def insert_data_fnames(data_path, labels_fname, label_file_columns, update_fn=None):
    '''Inserts the names of files not already in "data_path" into "labels_fname" csv file. Else, creates a labels file in csv format, using  "label_file_columns".. File contains column names in first row and data in the rest. This program inserts those names not already in "date__path" and applies update_fn (if provided) to the appended files. Prints the number of lines inserted. Returns None
    data_path: str, path of directory containing data files, relative to project root
    labels_fname: str, path of the csv file, relative to project root
    label_file_columns: list of str, ordered columns for the labels file'''
 
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
        df_append = update_fn(df_append, data_path)
 
    df_to_csv = pd.concat([df, df_append])
    df_to_csv.reset_index()
    df_to_csv.to_csv(labels_fname, mode='w', index=False, header=True)
    return None


def screenshot_height_width_update(df, screenshot_path):
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

    labels_df = pd.read_csv(sq_labels_fname)
    print(labels_df.columns)
    print(labels_df.head())
    
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
            labels_df.loc[(crnt_label_df.index)[i],'SqColor-BW'] = square_color_radios[i][0].get()
            labels_df.loc[(crnt_label_df.index)[i],'PcColor-BWE'] = piece_color_radios[i][0].get()
            labels_df.loc[(crnt_label_df.index)[i],'PcType-PRNBQK'] = piece_type_radios[i][0].get()
            labels_df.loc[(crnt_label_df.index)[i],'HumCheck-YN'] = human_check_radios[i][0].get()
        crnt_label_idx = last_label_idx
        last_label_idx = crnt_label_idx + min(label_ct, len(labels_df)-crnt_label_idx)
        print('crnt_label_idx: {0}\nlast_label_idx: {1}'.format(crnt_label_idx,last_label_idx))
        crnt_label_df = labels_df.iloc[crnt_label_idx:last_label_idx,:]
        print(type(crnt_label_df)) 
        print(crnt_label_df.shape) 
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

            insert_radios(square_colors, square_color_radios, i, col_sq_bgn,'SqColor-BW')
            insert_radios(piece_colors, piece_color_radios, i, col_clr_bgn, 'PcColor-BWE') 
            insert_radios(piece_types, piece_type_radios, i, col_typ_bgn, 'PcType-PRNBQK')
            insert_radios(human_check, human_check_radios, i, col_hum_bgn, 'HumCheck-YN') 

    draw_rows()     
    make_estimates_button = tkinter.Button(root, text="Make Estimates", command = on_make_estimates)
    make_estimates_button.grid(row=label_ct+1, column = 1, columnspan=2)
    mark_checked_button = tkinter.Button(root, text="Mark As Checked", command = on_mark_checked)
    mark_checked_button.grid(row=label_ct+1, column = 3, columnspan=3)
    next_button = tkinter.Button(root, text="Save Frame & Next", command = on_save_frame_and_next)
    next_button.grid(row=label_ct+1, column = 13, columnspan=2)
    root.mainloop()    
    print('end')

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

def update_screenshot_labels(screenshot_path, screenshot_labels_fname):
    '''Updates screenshot labels and boundingboxes
    screenshot_path: Relative path to the chess screenshot image directory
    screenshot_labels_fname: Relative path of the screenshot labels file
    '''

    if not os.path.exists(screenshot_labels_fname):
        print('File Name \'{0}\' does not exist.'.format(screenshot_labels_fname))
         
    if not os.path.exists(screenshot_path):
        print('Path \'{0}\' does not exist.'.format(screenshot_path))
    
    MAX_DISP_HEIGHT = 600
    MAX_DISP_WIDTH = 1000

    labels_df = pd.read_csv(screenshot_labels_fname)
    
    root = tkinter.Tk()
    root.title('Update Screenshot Labels')

    scr_fnames = list(set(labels_df['fname']))
    if len(scr_fnames)>0:
        scr_i = 0
   
    crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]

    print(crnt_scr_gr)
    canvas = tkinter.Canvas(root)
   
    row_ct = 0
    labels= []
    im_h_shrink, im_w_shrink = 1, 1
    def save_pos(event):
        global lastx, lasty
        lastx, lasty = event.x, event.y

    def addRectangle(event):
        nonlocal canvas, crnt_scr_gr
        #rect_id = canvas.create_rectangle((lastx, lasty, event.x, event.y), outline='white')
        #text_id = canvas.create_text((min(event.x, lastx), min(event.y,lasty)), text="Label1", fill='white', anchor="nw")
        _cols = ['label','x_min_pxl','y_min_pxl','x_max_pxl','y_max_pxl','HumCheck-YN']
        empty_processed = False
        for _i, _r in crnt_scr_gr.iterrows():
            if any(pd.isna(labels_df.loc[_i,_cols])) and not empty_processed:
                print('got empty')
                labels_df.loc[_i, _cols] = ['Chessboard', min(lastx, event.x), min(lasty, event.y), max(lastx, event.x), max(lasty, event.y), 'Y']
                crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]
                #crnt_scr_dict[_i] = [rect_id, text_id]
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
        print(im_w_new, im_h_new)
        im = im.resize((im_w_new, im_h_new))
        print(im.width, im.height)
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
        print("Row: {} deleted.".format(i))
        _cols = ['label','x_min_pxl','y_min_pxl','x_max_pxl','y_max_pxl','HumCheck-YN']
        if len(crnt_scr_gr)==1:
           for _c in _cols:
               labels_df.loc[i,_c] = np.nan 
        else:
           labels_df.drop(index=i, inplace=True)
        crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]
        print(crnt_scr_gr)
        draw_rows()
    
    def draw_rows():
        nonlocal row_ct, labels
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

    def on_save_frame_and_next():
        nonlocal scr_i, crnt_scr_gr
        labels_df.to_csv(screenshot_labels_fname, index=False)
    
        scr_fnames = list(set(labels_df['fname']))
        if scr_i<len(scr_fnames):
            scr_i += 1
   
        crnt_scr_gr = labels_df[labels_df['fname']==scr_fnames[scr_i]]

        draw_canvas()
        draw_labels()
        draw_header() 
        draw_rows()


    row_ct += 1 

    def draw_frame():
        pass

    def clear_frame():
        pass


    draw_canvas()
    draw_labels()
    draw_header() 
    draw_rows()
                
    next_button = tkinter.Button(root, text="Save Frame & Next", command = on_save_frame_and_next)
    next_button.grid(row=row_ct+2, column = 10, columnspan=1)

    root.mainloop()