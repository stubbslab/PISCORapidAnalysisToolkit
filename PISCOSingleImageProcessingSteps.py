import os
import SharedPISCOCommands as spc
import cantrips as c
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def overscanCorrect(original_image_file, target_dir, binning, verbose, shared_commands = None, img_already_split = 1, rm_prev_images = 1, oscan_prefix = None):
    if shared_commands == None:
        shared_commands = spc.CommandHolder()
    if oscan_prefix == None:
        oscan_prefix = shared_commands.getOverscanPrefix()
    if verbose: print ('Overscan correcting image ' + str(original_image_file))
    overscanned_files = shared_commands.overscanCorrect(original_image_file, target_dir = target_dir, binning = binning, img_already_split = img_already_split, verbose = 0)
    if rm_prev_images:
        [os.remove(target_dir + single_amp_image[len(oscan_prefix):]) for single_amp_image in overscanned_files ]
    #new_image_files = [oscan_prefix + single_amp_image for single_amp_image in image_files]
    return overscanned_files


def stitchImage(image_files, image_root, target_dir, verbose, shared_commands = None, single_filt_strs = None, rm_prev_images = 1):
    if shared_commands == None:
        shared_commands = spc.CommandHolder()
    if single_filt_strs == None:
        single_filt_strs = shared_commands.getFilters()
    filter_keyword = shared_commands.getFilterHeaderKeyword()
    images_data_and_header = [c.readInDataFromFitsFile(image_file, target_dir) for image_file in image_files]
    image_data_arrays = [image_data_and_header[0] for image_data_and_header in images_data_and_header]
    image_headers = [image_data_and_header[1] for image_data_and_header in images_data_and_header]
    stitched_image = shared_commands.PISCOStitch(image_data_arrays)
    headers = [image_headers[header_band_num * 2] for header_band_num in range(len(image_headers) // 2)]
    for header_band_num in range(len(headers)):
        headers[header_band_num][filter_keyword] = single_filt_strs[header_band_num]
    stitched_image_files = shared_commands.WriteStitchedImageToSingleBandFiles(stitched_image, headers, image_root, save_dir = target_dir)
    if rm_prev_images:
        [os.remove(target_dir + oscanned_flat) for oscanned_flat in image_files ]
    return stitched_image_files


def biasCorrect(image_files, target_dir, master_bias_files = None, master_bias_data_arrays = None, master_bias_data_headers = None, single_filt_strs = None, shared_commands = None, rm_prev_images = 1, verbose = 0):
    if shared_commands == None:
        shared_commands = spc.CommandHolder()
    if single_filt_strs == None:
        single_filt_strs = shared_commands.getFilters()
    image_extension = shared_commands.getImageExtension()
    filt_extensions = [shared_commands.getSingleBandSuffixPrefix() + filt_str for filt_str in single_filt_strs]
    bias_sub_keyword = shared_commands.getBiasSubKeyword()
    bias_sub_prefix = shared_commands.getBiasSubPrefix()

    if master_bias_data_arrays == None or master_bias_data_headers == None:
        if master_bias_files == None:
            master_bias_root = shared_commands.getMasterBiasRoot()
            master_bias_files = [master_bias_root + filt_extension + image_extension for filt_extension in filt_extensions]
        if verbose: print ('Loading bias images ' + str(master_bias_files))
        loaded_biases_and_headers = [c.readInDataFromFitsFile(bias_file, target_dir) for bias_file in master_bias_files]
        master_bias_data_arrays = [loaded_biases_and_headers[band_num][0] for band_num in range(len(single_filt_strs))]
        master_bias_data_headers = [loaded_biases_and_headers[band_num][1] for band_num in range(len(single_filt_strs))]

    data_to_bias_sub = [c.readInDataFromFitsFile(single_band_file, target_dir) for single_band_file in image_files]
    data_to_bias_sub_arrays = [data_to_bias_sub[band_num][0] for band_num in range(len(single_filt_strs))]
    data_to_bias_sub_headers = [data_to_bias_sub[band_num][1] for band_num in range(len(single_filt_strs))]
    bias_subbed_data = np.array(data_to_bias_sub_arrays) - np.array(master_bias_data_arrays)
    for header_num in range(len(data_to_bias_sub_headers)):
        data_to_bias_sub_headers[header_num][bias_sub_keyword] = 'Bias corrected using master bias ' + master_bias_files[header_num]
    bias_subbed_image_files = [bias_sub_prefix +  image_files[band_num] for band_num in range(len(bias_subbed_data))]
    [c.saveDataToFitsFile(np.transpose(bias_subbed_data[band_num]), bias_subbed_image_files[band_num], save_dir = target_dir, header = data_to_bias_sub_headers[band_num], ) for band_num in range(len(bias_subbed_data))]
    if rm_prev_images:
        [os.remove(target_dir + single_band_file) for single_band_file in image_files ]
    bias_subbed_image_files = bias_subbed_image_files

    return bias_subbed_image_files


def flatCorrect(image_files, target_dir, master_flat_files = None, master_flat_data_arrays = None, master_flat_data_headers = None, single_filt_strs = None, shared_commands = None, rm_prev_images = 1, verbose = 0):
    if shared_commands == None:
        shared_commands = spc.CommandHolder()
    if single_filt_strs == None:
        single_filt_strs = shared_commands.getFilters()
    image_extension = shared_commands.getImageExtension()
    filt_extensions = [shared_commands.getSingleBandSuffixPrefix() + filt_str for filt_str in single_filt_strs]
    flat_norm_keyword = shared_commands.getFlatNormKeyword()
    flat_norm_prefix = shared_commands.getFlatNormPrefix()

    if master_flat_data_arrays == None or master_flat_data_headers == None:
        if master_flat_files == None:
            master_flat_root = shared_commands.getMasterFlatRoot()
            master_flat_files = [master_flat_root + filt_extension + image_extension for filt_extension in filt_extensions]
        if verbose: print ('Loading flat images ' + str(master_flat_files))
        loaded_flats_and_headers = [c.readInDataFromFitsFile(flat_file, target_dir) for flat_file in master_flat_files]
        master_flat_data_arrays = [loaded_flats_and_headers[band_num][0] for band_num in range(len(single_filt_strs))]
        master_flat_data_headers = [loaded_flats_and_headers[band_num][1] for band_num in range(len(single_filt_strs))]

    data_to_flat_norm = [c.readInDataFromFitsFile(single_band_file, target_dir) for single_band_file in image_files]
    data_to_flat_norm_arrays = [data_to_flat_norm[band_num][0] for band_num in range(len(single_filt_strs))]
    data_to_flat_norm_headers = [data_to_flat_norm[band_num][1] for band_num in range(len(single_filt_strs))]
    flat_normed_data = np.array(data_to_flat_norm_arrays) / np.array(master_flat_data_arrays)
    for header_num in range(len(data_to_flat_norm_headers)):
        data_to_flat_norm_headers[header_num][flat_norm_keyword] = 'Flat fielded using master flat ' + master_flat_files[header_num]
    flat_norm_single_band_image_files = [flat_norm_prefix +  image_files[band_num] for band_num in range(len(flat_normed_data))]
    [c.saveDataToFitsFile(np.transpose(flat_normed_data[band_num]), flat_norm_single_band_image_files[band_num], save_dir = target_dir, header = data_to_flat_norm_headers[band_num]) for band_num in range(len(flat_normed_data))]
    if rm_prev_images:
        [os.remove(target_dir + single_band_file) for single_band_file in image_files ]
    new_image_files = flat_norm_single_band_image_files

    return new_image_files

def cropImage(image_files, target_dir, single_filt_strs = None, shared_commands = None, rm_prev_images = 1):
    if shared_commands == None:
        shared_commands = spc.CommandHolder()
    if single_filt_strs == None:
        single_filt_strs = shared_commands.getFilters()
    image_extension = shared_commands.getImageExtension()
    filt_extensions = [shared_commands.getSingleBandSuffixPrefix() + filt_str for filt_str in single_filt_strs]
    crop_sections = shared_commands.get1x1CropRegion()
    crop_prefix = shared_commands.getCropPrefix()
    crop_keyword = shared_commands.getCropKeyword()

    data_to_crop = [c.readInDataFromFitsFile(image_file, target_dir) for image_file in image_files]
    data_to_crop_arrays = [data_to_crop[band_num][0] for band_num in range(len(single_filt_strs))]
    data_to_crop_headers = [data_to_crop[band_num][1] for band_num in range(len(single_filt_strs))]

    cropped_data = [np.transpose(data_to_crop_arrays[band_num])[crop_sections[band_num][0][0]:crop_sections[band_num][0][1], crop_sections[band_num][1][0]:crop_sections[band_num][1][1]]  for band_num in range(len(single_filt_strs))]
    for header_num in range(len(data_to_crop_headers)):
        crop_section = crop_sections[header_num]
        data_to_crop_headers[header_num][crop_keyword] = 'Crop section is' + str(crop_section)
    cropped_single_band_image_files = [crop_prefix +  image_files[band_num] for band_num in range(len(cropped_data))]
    [c.saveDataToFitsFile(cropped_data[band_num], cropped_single_band_image_files[band_num], save_dir = target_dir, header = data_to_crop_headers[band_num]) for band_num in range(len(cropped_data))]
    if rm_prev_images:
        [ os.remove(target_dir + single_band_file) for single_band_file in image_files ]
    new_image_files = cropped_single_band_image_files

    return new_image_files

#We need to measure both a multiplicative and an additive term (first order polynomial fit, though the code can fit any order)
def correctDifferingAmpGains(image_files, target_dir,
                             shared_commands = None, single_filt_strs = None, binning = None, rm_prev_images = 1, amp_fit_order = 1,
                             sky_fit_order = 5, sig_clipping_for_bg_fit = 4, show_fit = 0):
    if shared_commands == None:
        shared_commands = spc.CommandHolder()
    if single_filt_strs == None:
        single_filt_strs = shared_commands.getFilters()
    if binning == None:
        binning = shared_commands.getBinning()
    image_extension = shared_commands.getImageExtension()
    filt_extensions = [shared_commands.getSingleBandSuffixPrefix() + filt_str for filt_str in single_filt_strs]

    amp_correct_col_range_1x1 = shared_commands.getAmpCorrectColRange( )
    amp_correct_pixel_threshold = shared_commands.getAmpGainPixelThreshold()
    fit_width = shared_commands.getAmpGainCorrectionWidth()
    sig_clip = shared_commands.getAmpGainCorrectionFitSigClip()
    amp_correction_fit_order = shared_commands.getAmpGainCorrectionFitOrder()
    gain_correct_prefix = shared_commands.getGainCorrectionPrefix()
    crop_sections = shared_commands.get1x1CropRegion()
    col_range_keyword = shared_commands.getAmpFitColRangeHeaderKeyword()
    gain_fit_keyword = shared_commands.getAmpFitHeaderKeyword()

    amp_correct_col_range = [col_range // binning for col_range in amp_correct_col_range_1x1 ]
    precrop_amp_junctures = [shared_commands.getOverscanSections1x1()[det_num*2][0] // binning for det_num in range(len(single_filt_strs))]
    amp_junctures = [precrop_amp_junctures[det_num] - crop_sections[det_num][0][0] for det_num in range(len(precrop_amp_junctures)) ]

    data_to_gain_correct = [c.readInDataFromFitsFile(single_band_file, target_dir) for single_band_file in image_files]
    data_to_gain_correct_arrays = [data_to_gain_correct[band_num][0] for band_num in range(len(single_filt_strs))]
    data_to_gain_correct_headers = [data_to_gain_correct[band_num][1] for band_num in range(len(single_filt_strs))]
    left_strips = [data_to_gain_correct_arrays[band_num][amp_correct_col_range[0]:amp_correct_col_range[1], amp_junctures[band_num] - fit_width:amp_junctures[band_num]] for band_num in range(len(single_filt_strs))]
    right_strips = [data_to_gain_correct_arrays[band_num][amp_correct_col_range[0]:amp_correct_col_range[1], amp_junctures[band_num]:amp_junctures[band_num] + fit_width] for band_num in range(len(single_filt_strs))]
    left_meds = [np.median(left_strip, axis = 1) for left_strip in left_strips]
    left_clipped_bg_fits = [c.polyFitNSigClipping(list(range(len(left_med))), left_med, sky_fit_order, sig_clipping_for_bg_fit) for left_med in left_meds]
    right_meds = [np.median(right_strip, axis = 1) for right_strip in right_strips]
    right_clipped_bg_fits = [c.polyFitNSigClipping(list(range(len(right_med))), right_med, sky_fit_order, sig_clipping_for_bg_fit) for right_med in right_meds]
    amp_ratios = [left_meds[band_num] / right_meds[band_num] for band_num in range(len(single_filt_strs))]
    clipped_means = [c.sigClipMean(amp_ratio, sig_clip = sig_clip) for amp_ratio in amp_ratios]
    clipped_amp_stds = [c.sigClipStd(amp_ratio, sig_clip = sig_clip) for amp_ratio in amp_ratios]

    included_rows_in_fits = [c.intersection(right_clipped_bg_fits[img_fit_num][0], left_clipped_bg_fits[img_fit_num][0]) for img_fit_num in range(len(left_clipped_bg_fits))]
    #print ('included_rows_in_fits = ' + str(included_rows_in_fits))
    old_included_rows_in_fits = included_rows_in_fits[:]
    #print ('[ [[left_meds[img_fit_num][row], right_meds[img_fit_num][row]] for row in included_rows_in_fits[img_fit_num]] for img_fit_num in range(len(included_rows_in_fits))] = ' + str([ [[left_meds[img_fit_num][row], right_meds[img_fit_num][row]] for row in included_rows_in_fits[img_fit_num]] for img_fit_num in range(len(included_rows_in_fits))]) )
    included_rows_in_fits = [[row for row in included_rows_in_fits[img_fit_num] if (left_meds[img_fit_num][row] < amp_correct_pixel_threshold and right_meds[img_fit_num][row] < amp_correct_pixel_threshold)] for img_fit_num in range(len(included_rows_in_fits))]
    #print ('included_rows_in_fits = ' + str(included_rows_in_fits))
    print ('[[row for row in old_included_rows_in_fits[img_fit_num] if not(row in included_rows_in_fits[img_fit_num])] for img_fit_num in range(len(included_rows_in_fits))] = ' + str([[row for row in old_included_rows_in_fits[img_fit_num] if not(row in included_rows_in_fits[img_fit_num])] for img_fit_num in range(len(included_rows_in_fits))]))
    left_cols_clipped = [[left_meds[img_fit_num][row] for row in included_rows_in_fits[img_fit_num]] for img_fit_num in range(len(included_rows_in_fits))]
    right_cols_clipped = [[right_meds[img_fit_num][row] for row in included_rows_in_fits[img_fit_num]] for img_fit_num in range(len(included_rows_in_fits))]
    mean_right_cols_clipped = [np.mean(right_col_clipped) for right_col_clipped in right_cols_clipped]
    mean_left_cols_clipped = [np.mean(left_col_clipped) for left_col_clipped in left_cols_clipped]
    #left_of_right_fits = [np.polyfit(np.array(right_cols_clipped[img_fit_num]) - mean_right_cols_clipped[img_fit_num], np.array(left_cols_clipped[img_fit_num]) - mean_left_cols_clipped[img_fit_num], amp_fit_order) for img_fit_num in range(len(left_cols_clipped))]
    #left_of_right_fits = [[left_of_right_fits[img_fit_num][0], left_of_right_fits[img_fit_num][1] + mean_left_cols_clipped[img_fit_num] - left_of_right_fits[img_fit_num][0] * mean_right_cols_clipped[img_fit_num]] for img_fit_num in range(len(left_of_right_fits))]
    left_of_right_fits = [c.doOrthogonalLinearFit(np.array(right_cols_clipped[img_fit_num]) - mean_right_cols_clipped[img_fit_num], np.array(left_cols_clipped[img_fit_num]) - mean_left_cols_clipped[img_fit_num], x_errs = None, y_errs = None, init_guess =[1.0, np.mean(left_cols_clipped[img_fit_num]) - np.mean(right_cols_clipped[img_fit_num]) ]).beta for img_fit_num in range(len(left_cols_clipped))]
    left_of_right_fits = [[left_of_right_fits[img_fit_num][0], left_of_right_fits[img_fit_num][1] + mean_left_cols_clipped[img_fit_num] - left_of_right_fits[img_fit_num][0] * mean_right_cols_clipped[img_fit_num]] for img_fit_num in range(len(left_of_right_fits))]
    print ('left_of_right_fits = ' + str(left_of_right_fits))
    if show_fit:
        f, axarr = plt.subplots(2,2, figsize = [8,8])
        for band_num in range(4):
            unclipped_scat = axarr[band_num // 2, band_num % 2].scatter(right_meds[band_num], left_meds[band_num], c = 'r', marker = '.')
            clipped_scat = axarr[band_num // 2, (band_num % 2)].scatter(right_cols_clipped[band_num], left_cols_clipped[band_num], c = 'g', marker = '.')
            axarr[band_num // 2, (band_num % 2)].set_xlim(min(right_cols_clipped[band_num]) * 0.95, max(right_cols_clipped[band_num]) * 1.05)
            axarr[band_num // 2, (band_num % 2)].set_ylim(min(left_cols_clipped[band_num]) * 0.95, max(left_cols_clipped[band_num]) * 1.05)
            perpFit_plot = axarr[band_num // 2, (band_num % 2)].plot(right_cols_clipped[band_num], np.poly1d(left_of_right_fits[band_num])(right_cols_clipped[band_num]), c = 'r')[0]
            axarr[band_num // 2, (band_num % 2) ].legend([unclipped_scat, clipped_scat, perpFit_plot],
                                                         ['Unclipped ','Clipped (fit)', 'fit: ' + ', '.join([str(c.round_to_n(fit_param, 4)) for fit_param in left_of_right_fits[band_num]])])
            axarr[band_num // 2, (band_num % 2) ].set_xlabel('Binned ADU along right amp')
            axarr[band_num // 2, (band_num % 2) ].set_ylabel('Binned ADU along left amp')
            axarr[band_num // 2, (band_num % 2) ].set_title(image_files[band_num])

        plt.tight_layout()
        plt.show()
    #x_fit_data = [[i for i in range(amp_correct_col_range[0], amp_correct_col_range[1]) if abs(amp_ratios[band_num][i - amp_correct_col_range[0]] - clipped_means[band_num]) < sig_clip * clipped_amp_stds[band_num]] for band_num in range(len(single_filt_strs))]
    #y_fit_data = [[elem for elem in amp_ratios[band_num] if abs(elem - clipped_means[band_num]) < sig_clip * clipped_amp_stds[band_num]] for band_num in range(len(single_filt_strs))]

    #amp_poly_fits = [np.polyfit(x_fit_data[band_num], y_fit_data[band_num], amp_correction_fit_order) for band_num in range(len(single_filt_strs))]

    #amp_fit_functs = [np.poly1d(amp_poly_fit) for amp_poly_fit in amp_poly_fits]
    for band_num in range(len(single_filt_strs)):
        print ('Working on band_num ' + str(band_num))
        amp_juncture = amp_junctures[band_num]
        amp_fit_funct = np.poly1d(left_of_right_fits[band_num])
        y_max, x_max = np.shape(data_to_gain_correct_arrays[band_num])
        x_mesh, y_mesh = np.meshgrid(np.arange(x_max), np.arange(y_max))
        data_to_gain_correct_arrays[band_num] = data_to_gain_correct_arrays[band_num] * (x_mesh < amp_juncture) + amp_fit_funct(data_to_gain_correct_arrays[band_num]) * (x_mesh >= amp_juncture)
        data_to_gain_correct_headers[band_num][col_range_keyword] = (str(amp_correct_col_range), 'Cols to determine amp correction')
        data_to_gain_correct_headers[band_num][gain_fit_keyword] = (str(left_of_right_fits[band_num]), 'Poly fit params of ratio of amp gains')

    gain_correct_single_band_image_files = [gain_correct_prefix + image_files[band_num] for band_num in range(len(data_to_gain_correct_arrays))]
    [c.saveDataToFitsFile(np.transpose(data_to_gain_correct_arrays[band_num]), gain_correct_single_band_image_files[band_num], save_dir = target_dir, header = data_to_gain_correct_headers[band_num]) for band_num in range(len(data_to_gain_correct_arrays))]
    if rm_prev_images:
        [os.remove(target_dir + single_band_file) for single_band_file in image_files ]
    new_image_files = gain_correct_single_band_image_files

    return new_image_files
