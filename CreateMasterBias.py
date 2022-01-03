import sys
import cantrips as c
import SharedPISCOCommands as spc
import os
import numpy as np


if __name__=="__main__":
    shared_commands = spc.CommandHolder()
    command_args = sys.argv[1:]
    if len(command_args) == 0:
        default_bias_list = shared_commands.getBiasList()
        print ('No commands passed.  Assuming biases are listed in file: "' + default_bias_list + '" ...' )
        bias_list = default_bias_list
    else:
        bias_list = command_args[0]
        print ('Setting bias list to "' + bias_list + '"')
    if os.path.exists('' + bias_list):
        print ('Will merge images ' + str(bias_images) + ' into master bias file.')
        if verbose: print('Starting to make master bias...')
        bias_images=np.loadtxt(bias_list, dtype='str')
        if len(np.shape(bias_images)) == 0:
            bias_images = [str(bias_images)]
        if verbose: print ('Overscan correcting individual biases...')
        for bias_image in bias_images:
            OverscanCorrect(bias_image, target_dir, binning, oscan_prefix = oscan_prefix)
            oscanned_biases = [ oscan_prefix + bias_image for bias_image in bias_images ]
            print ('Median combining overscanned images, in parts ...')
            med_bias = c.smartMedianFitsFiles(oscanned_biases, target_dir, bias_x_partitions, bias_y_partitions, n_mosaic_image_extensions = n_mosaic_image_extensions)
            if rm_oscan:
                [os.remove(target_dir + oscanned_bias) for oscanned_bias in oscanned_biases ]
            if verbose: print ('Stitching median of biases')
            bias_images=PISCOStitch(med_bias)
            WriteFITSFB(bias_images, target_dir + outputbiasfilename)
    else:
        master_bias_name = shared_commands.getMasterBias()
        shared_cal_dir = shared_commands.getSharedCalDir()
        print ('Could not find bias list ' + bias_list)
        print ('Looking for master bias file ' + master_bias_name + ' in directory ' + shared_cal_dir)
