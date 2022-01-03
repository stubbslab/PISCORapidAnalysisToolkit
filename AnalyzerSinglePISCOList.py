import PISCOReducerClass as prc
import PISCOAnalyzerClass as pac
import sys
import numpy as np

if __name__ == "__main__":
    sysargs = sys.argv[1:]
    print ('sysargs = ' + str(sysargs))
    science_list, target_dir, create_bias, create_flat, rm_partial_files, verbose = sysargs
    create_bias = int(create_bias)
    create_flat = int(create_flat)
    rm_partial_files = int(rm_partial_files)
    verbose = int(verbose)

    print ('[science_list, target_dir, create_bias, create_flat, rm_partial_files, verbose] = ' + str([science_list, target_dir, create_bias, create_flat, rm_partial_files, verbose]))

    pisco_reducer = prc.PISCOReducer(target_dir, rm_partial_calib_images = rm_partial_files)

    pisco_analyzer = pac.PISCOAnalyzer(target_dir, rm_intermediate_files = rm_partial_files)

    if create_bias:
        pisco_reducer.createMasterBias()

    if create_flat:
        pisco_reducer.createMasterFlat()

    image_files = np.loadtxt(target_dir + science_list, dtype='str')

    fully_processed_images = [[] for image_file in image_files]

    for image_file_num in range(len(image_files)):
        image_file = image_files[image_file_num]
        print ('Starting reduction of science image file ' + str(image_file))
        reduced_single_band_images = pisco_reducer.fullReduceImage(image_file, bias_correct = 0, flat_correct = 0, crop_image = 1, gain_correct = 1,
                                                                    split_image = 1, oscan = 1, stitch_image = 1, verbose = verbose, )
        wcs_single_band_images = [pisco_analyzer.computeWCSForImage(single_band_image) for single_band_image in reduced_single_band_images]
        fully_processed_images[image_file_num] = wcs_single_band_images
