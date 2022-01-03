import sys
import cantrips as c
import SharedPISCOCommands as spc
import os
import numpy as np


def creatMasterBias(verbose = 1, overscan_correct = 1, rm_overscan = 1, target_dir = ''):
    shared_commands = spc.CommandHolder()
    overscan_prefix = shared_commands.getOverscanPrefix()
    bias_x_partitions = shared_commands.getBiasXPartitions()
    bias_y_partitions = shared_commands.getBiasYPartitions()
    n_pisco_mosaics = shared_commands.getNMosaics()
    master_bias_root = shared_commands.getMasterBiasRoot()
    command_args = sys.argv[1:]
    if len(command_args) == 0:
        default_bias_list = shared_commands.getBiasList()
        print ('No commands passed.  Assuming biases are listed in file: "' + default_bias_list + '" ...' )
        bias_list = default_bias_list
    else:
        bias_list = command_args[0]
        print ('Setting bias list to "' + bias_list + '"')
    if os.path.exists('' + bias_list):
        bias_images=np.loadtxt(bias_list, dtype='str')
        print ('Will merge images ' + str(bias_images) + ' into master bias file.')
        if verbose: print('Starting to make master bias...')
        if len(np.shape(bias_images)) == 0:
            bias_images = [str(bias_images)]
        if verbose: print ('Overscan correcting individual biases...')
        if overscan_correct:
            for bias_image in bias_images:
                shared_commands.overscanCorrect(bias_image, verbose = verbose)
            biases_to_stitch = [ overscan_prefix + bias_image for bias_image in bias_images ]
        else:
            biases_to_stitch =  bias_images[:]
        for band_num bias_image in bias_to_stitch:
            shared_commands.piscoCrop(bias_image, verbose = verbose)
        biases_to_stitch = [crop_prefix + bias_image for bias_image in biases_to_stitch]
        print ('Median combining overscanned images, in parts ...')
        med_bias = c.smartMedianFitsFiles(biases_to_stitch, target_dir, bias_x_partitions, bias_y_partitions, n_mosaic_image_extensions = n_pisco_mosaics )
        _, header = c.readInDataFromFitsFile(biases_to_stitch[0], target_dir)
        header[shared_commands.getNCombinedHeaderKeyword()] = len(biases_to_stitch)
        if rm_overscan :
            [os.remove(overscan_prefix + bias_image) for bias_image in bias_images  ]
        if verbose: print ('Stitching median of biases')
        med_bias = shared_commands.PiscoCrop(self, image, band_num, header)
        bias_images = shared_commands.PISCOStitch(med_bias)
        shared_commands.WriteStitchedImageToSingleBandFiles(med_bias, [header for bias_image in bias_images], target_dir + master_bias_root)
    else:
        shared_cal_dir = shared_commands.getSharedCalDir()
        print ('Could not find bias list ' + bias_list)
        print ('Looking for master bias file ' + master_bias_root + ' in directory ' + shared_cal_dir)
