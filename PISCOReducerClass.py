import sys
import cantrips as c
import SharedPISCOCommands as spc
import os
import numpy as np
import PISCOSingleImageProcessingSteps as singleImgFuncts


class PISCOReducer:

    def initializeMasterBias(self, master_bias_files, verbose = None):
        if verbose == None:
            verbose = self.verbose
        if master_bias_files == None:
            master_bias_root = self.shared_commands.getMasterBiasRoot()
            master_bias_files = [master_bias_root + filt_extension + self.image_extension for filt_extension in self.filt_extensions]
        self.master_bias_files = master_bias_files
        if verbose: print ('Loading bias images ' + str(master_bias_files))
        loaded_biases_and_headers = [c.readInDataFromFitsFile(bias_file, self.target_dir) for bias_file in self.master_bias_files]
        self.master_bias_data_arrays = [loaded_biases_and_headers[band_num][0] for band_num in range(len(self.single_filt_strs))]
        self.master_bias_data_headers = [loaded_biases_and_headers[band_num][1] for band_num in range(len(self.single_filt_strs))]
        return 1

    def initializeMasterFlat(self, master_flat_files, verbose = None):
        if verbose == None:
            verbose = self.verbose
        if master_flat_files == None:
            master_flat_root = self.shared_commands.getMasterFlatRoot()
            master_flat_files = [master_flat_root + filt_extension + self.image_extension for filt_extension in self.filt_extensions]
        self.master_flat_files = master_flat_files
        if verbose: print ('Loading flat images ' + str(self.master_flat_files))
        loaded_flats_and_headers = [c.readInDataFromFitsFile(flat_file, self.target_dir) for flat_file in self.master_flat_files]
        self.master_flat_data_arrays = [loaded_flats_and_headers[band_num][0] for band_num in range(len(self.single_filt_strs))]
        self.master_flat_data_headers = [loaded_flats_and_headers[band_num][1] for band_num in range(len(self.single_filt_strs))]

        return 1

    #Y
    def processSingleImage(self, image_file, bias_correct, flat_correct, crop_image, gain_correct,
                           split_image = 1, oscan = 1, stitch_image = 1,
                           target_dir = None, rm_intermediate_images = None,
                           verbose = None, master_bias_files = None, master_flat_files = None,
                           sky_fit_order = 3):
        if verbose: print ('Reducing image ' + str(image_file) + '...' )
        if target_dir == None:
            target_dir = self.target_dir
        if rm_intermediate_images == None:
            rm_intermediate_images = self.rm_partial_calib_images
        if verbose == None:
            verbose = self.verbose
        image_root = image_file[0:-len(self.image_extension)]

        #Split the raw image into individual amp segments.
        if split_image:
            if verbose: print('Splitting mosaic image into single amp images')
            current_single_amp_image_files = self.shared_commands.rawSplitImageFile(image_file, target_dir, n_mosaic_extensions = self.shared_commands.getNMosaics(), verbose = verbose)
        else:
            current_single_amp_image_files = image_file
        #Overscan correct
        if oscan:
            if verbose: print('Overscan correcting images')
            current_single_amp_image_files = singleImgFuncts.overscanCorrect(image_file, self.target_dir, self.binning, 0, shared_commands = self.shared_commands, img_already_split = 1, rm_prev_images = rm_intermediate_images)
            image_root = self.oscan_prefix + image_root
        print ('image_root = ' + str(image_root))

        #Stitch the image.  Now we are dealing with images by band, rather than by amp
        if stitch_image:
            print ('current_single_amp_image_files = ' + str(current_single_amp_image_files))
            if verbose: print('Stitching single amp images into single filter images')
            current_single_band_image_files = singleImgFuncts.stitchImage(current_single_amp_image_files, image_root, target_dir, verbose, shared_commands = self.shared_commands, single_filt_strs = self.single_filt_strs, rm_prev_images = rm_intermediate_images)
        else:
            current_single_band_image_files = current_single_amp_image_files
        print ('current_single_band_image_files = ' + str(current_single_band_image_files))
        #subtract the bias image
        if bias_correct:
            if verbose: print ('Bias correcting images...')
            if self.master_bias_data_arrays == None or self.master_bias_data_headers == None:
                self.initializeMasterBias(master_bias_files, verbose = verbose)
            current_single_band_image_files = singleImgFuncts.biasCorrect(current_single_band_image_files, target_dir, master_bias_files = self.master_bias_files, master_bias_data_arrays = self.master_bias_data_arrays , master_bias_data_headers = self.master_bias_data_headers, single_filt_strs = self.single_filt_strs, shared_commands = self.shared_commands, rm_prev_images = rm_intermediate_images, verbose = verbose)

        #apply flat field correction
        if flat_correct:
            if verbose: print ('Flat fielding images...')
            if self.master_flat_data_arrays == None or self.master_flat_data_headers == None:
                self.initializeMasterFlat(master_flat_files, verbose = verbose)
            current_single_band_image_files = singleImgFuncts.flatCorrect(current_single_band_image_files, target_dir, master_flat_files = self.master_flat_files, master_flat_data_arrays = self.master_flat_data_arrays , master_flat_data_headers = self.master_flat_data_headers, single_filt_strs = self.single_filt_strs, shared_commands = self.shared_commands, rm_prev_images = rm_intermediate_images, verbose = verbose)

        #Crop the single band images
        if crop_image:
            if verbose: print ('Cropping images...')
            current_single_band_image_files = singleImgFuncts.cropImage(current_single_band_image_files, target_dir, single_filt_strs = self.single_filt_strs, shared_commands = self.shared_commands, rm_prev_images = rm_intermediate_images)

        #Correct differing gains on different amplifiers on single ccd.
        if gain_correct:
            if verbose: print ('Matching gain ratios across amplifiers on single band images ...')
            current_single_band_image_files = singleImgFuncts.correctDifferingAmpGains(current_single_band_image_files, target_dir, shared_commands = self.shared_commands, single_filt_strs = self.single_filt_strs, rm_prev_images = rm_intermediate_images, sky_fit_order = sky_fit_order)
        if verbose: print ('Image ' + str(image_file) + ' successfully reduced.  Reduced files are: ' + str(current_single_band_image_files) )
        return current_single_band_image_files

    def createMasterBias(self, bias_list = None, oscan = None, target_dir = None,
                         x_partitions = None, y_partitions = None,
                         verbose = None, master_bias_root = None, rm_intermediate_images = None):
        if x_partitions == None:
            x_partitions = self.x_partitions
        if y_partitions == None:
            y_partitions = self.y_partitions
        if oscan == None:
            oscan = self.oscan

        if target_dir == None:
            target_dir = self.target_dir
        if rm_intermediate_images == None:
            rm_intermediate_images = self.rm_partial_calib_images
        if master_bias_root == None:
            master_bias_root = self.shared_commands.getMasterBiasRoot()
        if bias_list == None:
            list_extension = self.shared_commands.getListSuffix()
            bias_list = master_bias_root + list_extension

        print ('[self.target_dir, bias_list] = ' + str([self.target_dir, bias_list]))
        if os.path.exists(self.target_dir + bias_list):
            print ('Will merge images in list ' + str(bias_list) + ' into master bias file.')
            if verbose: print('Starting to make master bias...')
            bias_images=np.loadtxt(self.target_dir + bias_list, dtype='str')
            #First, we should split the mosaic image into pieces
            if len(np.shape(bias_images)) == 0:
                bias_images = [str(bias_images)]

            biases_to_stack = [self.processSingleImage(bias_image, 0, 0, 0, 0, target_dir = target_dir,
                                                       oscan = oscan, rm_intermediate_images = rm_intermediate_images,
                                                       verbose =verbose ) for bias_image in bias_images]
            print ('biases_to_stack = ' + str(biases_to_stack))

            print ('Median combining bias images, in parts ...')

            med_biases_arrays_and_headers = [ c.smartMedianFitsFiles([bias_to_stack[amp_num] for bias_to_stack in biases_to_stack], target_dir, x_partitions, y_partitions) for amp_num in range(len(biases_to_stack[0])) ]
            med_bias_arrays = [med_bias[0] for med_bias in med_biases_arrays_and_headers]
            med_bias_headers = [med_bias[1] for med_bias in med_biases_arrays_and_headers]
            print ('[len(med_bias_arrays), len(med_bias_headers)] = ' + str([len(med_bias_arrays), len(med_bias_headers)]))
            if verbose: print ('Stitching median of biases')
            #master_bias_image = self.shared_commands.PISCOStitch(med_bias_arrays)
            master_bias_files = self.shared_commands.WriteStitchedImageToSingleBandFiles((med_bias_arrays), med_bias_headers, master_bias_root, save_dir = target_dir)
            self.master_bias_data_arrays, self.master_bias_data_headers = [med_bias_arrays, med_bias_headers]
            self.master_bias_files = master_bias_files

            if rm_intermediate_images:
                [[os.remove(target_dir + single_amp_image) for single_amp_image in bias_to_stack ] for bias_to_stack in biases_to_stack]
        else:
            master_bias_name = self.shared_commands.getMasterBias()
            shared_cal_dir = self.shared_commands.getSharedCalDir()
            print ('Could not find bias list ' + bias_list)
            print ('Looking for master bias file ' + master_bias_name + ' in directory ' + shared_cal_dir)

    def PISCOSortFlatImages(self, flat_list_file, target_dir = None):
        if target_dir == None:
            target_dir = self.target_dir
        count_target_ranges = self.shared_commands.getGoodFlatADUs()
        amp_measure_section_1x1 = self.shared_commands.getFlatSingleAmpMeasureSections1x1()
        binning = self.shared_commands.getBinning()
        flat_file_names = self.shared_commands.getSingleBandFlatLists()
        verbose = self.verbose

        amp_measure_sections = (np.array(amp_measure_section_1x1) // binning).tolist()
        imgs_set = [ [], [], [], [] ]
        flatimages=np.loadtxt(target_dir + flat_list_file, dtype='str')
        if len(np.shape(flatimages)) < 1:
            flatimages=[str(flatimages)]
        allflat=[]
        for flat in flatimages:
            #allflat.append(PISCOStitch(loadimage(flat)[0]))
            amplifiers_data, headers = c.readInDataFromFitsFile(flat, target_dir, n_mosaic_image_extensions = self.shared_commands.getNMosaics() )
            amplifiers_data = np.array(amplifiers_data)
            if verbose: print ('Examining flat image ' + str(flat) + ' ...')
            for i in range(len(amplifiers_data) // 2):
                count_target_range = count_target_ranges[i]
                left_amp_measure_section = amp_measure_sections[2*i]
                left_amp_median = np.median(np.array(amplifiers_data[2*i,left_amp_measure_section[1][0]:left_amp_measure_section[1][1],
                                                                         left_amp_measure_section[0][0]:left_amp_measure_section[0][1] ]))
                right_amp_measure_section = amp_measure_sections[2*i+1]
                right_amp_median = np.median(np.array(amplifiers_data[2*i + 1, right_amp_measure_section[1][0]:right_amp_measure_section[1][1],
                                                                  right_amp_measure_section[0][0]:right_amp_measure_section[0][1] ]))

                average_amp_val = (left_amp_median + right_amp_median) / 2.0
                #if verbose: print ('[left_amp_median, right_amp_median, average_amp_val] = ' + str([left_amp_median, right_amp_median, average_amp_val]))
                if average_amp_val < count_target_range[1] and average_amp_val > count_target_range[0]:
                    if verbose: print ('Adding ' + flat + ' to set for flat file list ' + str(flat_file_names[i]))
                    imgs_set[i] = imgs_set[i] + [flat]

        for i in range(len(imgs_set)):
            print ('Saving images ' + str(imgs_set[i]) + ' to file ' + target_dir + flat_file_names[i])
            c.saveListToFile(imgs_set[i], flat_file_names[i], save_dir = target_dir, sep = '')
        if verbose: print ('Flats sorted into by-filter lists. ')

        return flat_file_names

    def createMasterFlat(self, flat_list = None, oscan = None, rm_oscan = None,
                         flat_x_partitions = None, flat_y_partitions = None, target_dir = None,
                         verbose = None, master_flat_root = None, rm_intermediate_images = None,
                         bias_correct = 1, master_bias_files = None   ):
        oscan_prefix = self.shared_commands.getOverscanPrefix()
        bias_sub_prefix = self.shared_commands.getBiasSubPrefix()
        bias_sub_keyword = self.shared_commands.getBiasSubKeyword()
        norm_sect_keyword = self.shared_commands.getNormSectKeyword()
        norm_prefix = self.shared_commands.getNormalizationPrefix()
        if oscan == None:
            oscan = self.oscan
        if flat_x_partitions == None:
            flat_x_partitions = self.x_partitions
        if flat_y_partitions == None:
            flat_y_partitions = self.y_partitions
        if verbose == None:
            verbose = self.verbose
        if target_dir == None:
            target_dir = self.target_dir
        if rm_intermediate_images == None:
            rm_intermediate_images = self.rm_partial_calib_images
        if master_flat_root == None:
            master_flat_root = self.shared_commands.getMasterFlatRoot()
        if flat_list == None:
            default_flat_list = self.shared_commands.getFlatList()
            print ('No new flat list file given.  Assuming flats are listed in PISCO defaults file: "' + default_flat_list + '" ...' )
            flat_list = default_flat_list
        else:
            print ('Setting flat list to "' + flat_list + '"')

        image_extension = self.shared_commands.getImageExtension()
        if os.path.exists(self.target_dir + flat_list):
            filt_strs = self.shared_commands.getFilters()
            filt_extensions = self.filt_extensions
            flat_norm_sections = self.shared_commands.getFlatStitchedMeasureSections1x1()
            n_bands = len(filt_strs)

            if verbose: print('Starting to make master flat...')
            single_band_flat_lists = self.PISCOSortFlatImages(flat_list)

            #processed_single_flats = []
            original_to_stackable_images_dict = {}
            for band_num in range(len(single_band_flat_lists)):
                flats_to_stack = []
                single_band_flat_list = single_band_flat_lists[band_num]

                filt_str = filt_strs[band_num]
                if verbose: print ('Will merge images in list ' + str(single_band_flat_list) + ' into master flat file for band ' + filt_str + '...')
                flat_images = np.loadtxt(self.target_dir + single_band_flat_list, dtype='str')
                if len(np.shape(flat_images)) == 0:
                    flat_images = [str(flat_images)]

                split_flat_images = [[] for flat_image in flat_images]
                single_flat_image_headers = [[] for flat_image in flat_images]
                single_band_flat_files_to_stack = []
                print ('flat_images = ' +str(flat_images))
                for flat_image in flat_images:
                    if not(flat_image in original_to_stackable_images_dict.keys()):
                        new_flat_images = self.processSingleImage(flat_image, bias_correct, 0, 0, 0,
                                                                  target_dir = target_dir,
                                                                  oscan = oscan, rm_intermediate_images = rm_intermediate_images,
                                                                  verbose = verbose, master_bias_files = master_bias_files )
                        print ('new_flat_images = ' + str(new_flat_images))
                        for single_band_num in range(len(new_flat_images)):
                            new_flat_image_data, new_flat_image_header = c.readInDataFromFitsFile(new_flat_images[single_band_num], target_dir)
                            flat_norm_section = flat_norm_sections[single_band_num]
                            new_flat_image_data = new_flat_image_data / np.median(new_flat_image_data[flat_norm_section[0][0]:flat_norm_section[0][1], flat_norm_section[1][0]:flat_norm_section[1][1]])
                            new_flat_image_header[norm_sect_keyword] = str(flat_norm_section) + ' (no cropping)'
                            c.saveDataToFitsFile(np.transpose(new_flat_image_data), norm_prefix + new_flat_images[single_band_num], target_dir, header = new_flat_image_header)
                            if rm_intermediate_images:
                                os.remove(target_dir + new_flat_images[single_band_num])
                        new_flat_images = [norm_prefix + new_flat_image for new_flat_image in new_flat_images ]
                        original_to_stackable_images_dict[flat_image] = new_flat_images
                    print ('flat_image = ' + str(flat_image))
                    flats_to_stack = flats_to_stack + [original_to_stackable_images_dict[flat_image]]
                print ('flats_to_stack = ' + str(flats_to_stack))

                median_flat_image, median_header = c.smartMedianFitsFiles([single_flat_file[band_num] for single_flat_file in flats_to_stack], self.target_dir, flat_x_partitions, flat_y_partitions)

                c.saveDataToFitsFile(np.transpose(median_flat_image), master_flat_root + self.shared_commands.getSingleBandSuffixPrefix() + self.single_filt_strs[band_num] + image_extension, self.target_dir, header = median_header)

                #print ('flats_to_stack = ' + str(flats_to_stack))

            if rm_intermediate_images:
                [ [os.remove(target_dir + single_band_image) for single_band_image in original_to_stackable_images_dict[original_image] ] for original_image in original_to_stackable_images_dict.keys() ]

    def fullReduceImage(self, image,
                        bias_correct = 1, flat_correct = 1, crop_image = 1, gain_correct = 1,
                        split_image = 1, oscan = 1, stitch_image = 1,
                        target_dir = None, rm_intermediate_images = None,
                        verbose = 1, master_bias_files = None, master_flat_files = None):

        processed_image_files = self.processSingleImage(image, bias_correct, flat_correct, crop_image, gain_correct,
                                                        split_image = split_image, oscan = oscan, stitch_image = stitch_image,
                                                        target_dir = target_dir, rm_intermediate_images = rm_intermediate_images,
                                                        verbose = verbose, master_bias_files = master_bias_files, master_flat_files = master_flat_files)

        self.reduced_images = self.reduced_images + [image]

        return processed_image_files

    def __init__(self, target_dir, oscan = 1, x_partitions = 4, y_partitions = 4, verbose = 1, rm_partial_calib_images = 1,):
        self.target_dir = target_dir
        self.shared_commands = spc.CommandHolder()
        self.oscan = oscan
        self.oscan_prefix = self.shared_commands.getOverscanPrefix()
        self.binning = self.shared_commands.getBinning()
        self.x_partitions, self.y_partitions = [x_partitions, y_partitions]
        self.verbose = verbose
        self.rm_partial_calib_images = rm_partial_calib_images
        self.single_filt_strs = self.shared_commands.getFilters()
        self.filt_extensions = [self.shared_commands.getSingleBandSuffixPrefix() + filt_str for filt_str in self.single_filt_strs]

        self.image_extension = self.shared_commands.getImageExtension()
        self.master_bias_root = self.shared_commands.getMasterBiasRoot()
        self.master_bias_file = self.master_bias_root + self.image_extension
        self.master_flat_root = self.shared_commands.getMasterFlatRoot()
        self.single_band_suffixes = [self.shared_commands.getSingleBandSuffixPrefix() + filt_str for filt_str in self.single_filt_strs]
        self.master_flat_files = [self.master_flat_root + suffix + self.image_extension for suffix in self.single_band_suffixes]
        self.master_bias_data_arrays = None
        self.master_bias_data_headers = None
        self.master_flat_data_arrays = None
        self.master_flat_data_headers = None
        self.reduced_images = []
