import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from utils_calcium import x_right_shift, y_down_shift, x_left_shift, y_up_shift, save_data, rotate_plus, rotate_minus, scale_minus, scale_plus
import numpy as np


#
def on_mouse_click(event):

    global calcium_object
    
    if event.button == 1:  # Check if left mouse button (button 1) is clicked

        if event.inaxes == calcium_object.ax:    

            # figure out which quadrant you are in relative to [0,512] and [0,512] plot and x,y centre coordinates

            if event.xdata < calcium_object.theta_x and event.ydata < calcium_object.theta_y:
                print ("bottom left quadrant")
            elif event.xdata < calcium_object.theta_x and event.ydata > calcium_object.theta_y:
                print ("top left quadrant")
            elif event.xdata > calcium_object.theta_x and event.ydata < calcium_object.theta_y:   
                print ("bottom right quadrant")
            elif event.xdata > calcium_object.theta_x and event.ydata > calcium_object.theta_y:
                print ("top right quadrant")
                
            # plot dashed axies

    if event.button == 2:  # Check if middel button pressed
        print("Middle button pressed")

        if event.inaxes == calcium_object.ax:
            x, y = event.xdata, event.ydata

            #
            print("Setting centre to: ", x, y)

            #
            calcium_object.theta_x = x
            calcium_object.theta_y = y

            #
            calcium_object.scale_x = x
            calcium_object.scale_y = y

            calcium_object.alignment_logger.append(['scale', 0.999])

            calcium_object.plot_quadrants()

#
def align_gui_local(ca_object):

    global calcium_object

    calcium_object = ca_object

    #
    calcium_object.x_shift = 0 
    calcium_object.y_shift = 0
    calcium_object.theta = 0
    calcium_object.theta_x = 256
    calcium_object.theta_y = 256
    calcium_object.scale_x = 256
    calcium_object.scale_y = 256
    calcium_object.scale_factor = 1
    calcium_object.n_cells_show = 200


    #
    calcium_object.cell_idxs = np.random.choice(len(calcium_object.sessions[calcium_object.session_selected].contours), 
                                size=min(calcium_object.n_cells_show, 
                                            len(calcium_object.sessions[calcium_object.session_selected].contours)), 
                                replace=False)
    calcium_object.day_cell_idx = np.random.choice(len(calcium_object.sessions[0].contours),
                                        size=min(calcium_object.n_cells_show, 
                                                len(calcium_object.sessions[0].contours)),
                                        replace=False)

    #
    calcium_object.alignment_logger = []

    ########################################
    ########################################
    ########################################
    #
    fig = plt.figure(figsize=(10,10))

    # Connect the mouse click event to the callback function
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    #
    calcium_object.ax = plt.subplot(1,1,1)

    #
    calcium_object.session_id = 0
    clr = 'red'
    session_id=0
    calcium_object.plot_session_contours(clr)

    #
    calcium_object.session_id = calcium_object.session_selected
    clr = 'blue'
    calcium_object.plot_session_contours(clr)

    #
    plt.legend()

    #
    plt.xlim(0,512)
    plt.ylim(0,512)

    # Create a button widget
    button_ax = plt.axes([0.12, 0.04, 0.05, 0.03])
    button = Button(button_ax, 'x_right')
    button.on_clicked(lambda event: x_right_shift(event, calcium_object))

    #
    button_ax1 = plt.axes([0.05, 0.04, 0.05, 0.03])
    button1 = Button(button_ax1, 'x_left')
    button1.on_clicked(lambda event: x_left_shift(event, calcium_object))

    #
    button_ax2 = plt.axes([0.08, 0.07, 0.05, 0.03])
    button2 = Button(button_ax2, 'y_up')
    button2.on_clicked(lambda event: y_up_shift(event, calcium_object))

    #
    button_ax3 = plt.axes([0.08, 0.01, 0.05, 0.03])
    button3 = Button(button_ax3, 'y_down')
    button3.on_clicked(lambda event: y_down_shift(event, calcium_object))

    #
    button_ax4 = plt.axes([0.8, 0.01, 0.05, 0.03])
    button4 = Button(button_ax4, 'save')
    button4.on_clicked(lambda event: save_data(event, calcium_object))

    #
    button_ax5 = plt.axes([0.64, 0.01, 0.05, 0.03])
    button5 = Button(button_ax5, 'rotate +')
    button5.on_clicked(lambda event: rotate_plus(event, calcium_object))

    #
    button_ax6 = plt.axes([0.7, 0.01, 0.05, 0.03])
    button6 = Button(button_ax6, 'rotate -')
    button6.on_clicked(lambda event: rotate_minus(event, calcium_object))

    #
    button_ax7 = plt.axes([0.4, 0.01, 0.05, 0.03])
    button7 = Button(button_ax7, 'scale - ')
    button7.on_clicked(lambda event: scale_minus(event, calcium_object))

    button_ax8 = plt.axes([0.46, 0.01, 0.05, 0.03])
    button8 = Button(button_ax8, 'scale + ')
    button8.on_clicked(lambda event: scale_plus(event, calcium_object))

    # 
    plt.show()