#After flat field and bias correcting images, there is a residual
# difference between adjacent amplifiers.  I need to determine if
# that effect is multiplicative (probably and unstable gain) or
# additive (probably an unstable bias level).
#My plan to do that is to measure the best fit ratio and linear
# offset for a range of flux images.  One should be stable,
# the other should vary with flux level.

import cantrips as c
import PISCOReducerClass as prc
import SharedPISCOCommands as spc
import numpy as np
import matplotlib.pyplot as plt

class AmplifierOffsetMeasurer:

    def measureAmpProperties(self, files_to_process, bias_correct = 1, flat_correct = 1, crop_image = 1, gain_correct = 0, split_image = 1, oscan = 1, stitch_image = 1, rm_partial_images = 1, filts_to_process = 'all', amp_fit_order = 1, sky_fit_order = 3, sig_clipping_for_bg_fit = 5, sig_clipping_for_stats = 3):

        if filts_to_process == 'all':
            filts_to_process = self.filt_strs
        indeces_to_process = [index for index in range(len(filts_to_process)) if self.filt_strs[index] in filts_to_process]
        processed_files_names = [['' for filt_to_process in filts_to_process] for files_to_process in files_to_process]

        for file_num in range(len(files_to_process)):
            file_to_process = files_to_process[file_num]
            processed_files = self.reducer.fullReduceImage(file_to_process, bias_correct = bias_correct, flat_correct = flat_correct, crop_image = crop_image, gain_correct = gain_correct, split_image = split_image, oscan = oscan, stitch_image = stitch_image, rm_intermediate_images = rm_partial_images)
            files_to_analyze = [processed_files[index] for index in indeces_to_process]
            for index_to_process in indeces_to_process:
                file_to_analyze = processed_files[index_to_process]
                print ('file_to_analyze = ' + str(file_to_analyze))
                processed_data_array, processed_data_header = c.readInDataFromFitsFile(file_to_analyze, self.target_dir)
                amp_juncture = self.amp_junctures[index_to_process]
                left_col = processed_data_array[:, (amp_juncture- self.amp_sample_width):amp_juncture]
                left_bin_col = np.median(left_col, axis = 1)
                left_clipped_rows, left_clipped_row_vals, _, left_col_fit = c.polyFitNSigClipping(list(range(len(left_bin_col))), left_bin_col, sky_fit_order, sig_clipping_for_bg_fit)
                #left_col_std = np.std(left_bin_col - np.poly1d(left_col_fit)(list(range(len(left_bin_col)))))

                right_col = processed_data_array[:, amp_juncture:amp_juncture + self.amp_sample_width]
                right_bin_col = np.median(right_col, axis = 1)
                right_clipped_rows, right_clipped_row_vals, _, right_col_fit  = c.polyFitNSigClipping(list(range(len(right_bin_col))), right_bin_col, sky_fit_order, sig_clipping_for_bg_fit)
                f, axarr = plt.subplots(2,1)
                axarr[0].plot(list(range(len(right_bin_col))), right_bin_col, c = 'g')
                axarr[0].plot(list(range(len(right_bin_col))), np.poly1d(right_col_fit)(list(range(len(right_bin_col)))), c = 'k')
                axarr[1].plot(list(range(len(left_bin_col))), left_bin_col, c = 'g')
                axarr[1].plot(list(range(len(left_bin_col))), np.poly1d(left_col_fit)(list(range(len(left_bin_col)))), c = 'k')
                plt.show()
                ratio = left_bin_col / right_bin_col
                ratio_fit = np.polyfit(list(range(len(ratio))), ratio, amp_fit_order)
                difference = left_bin_col - right_bin_col
                difference_fit = np.polyfit(list(range(len(difference))), difference, amp_fit_order)
                #left_col_std = np.std(left_bin_col)
                #left_col_median = np.median(left_bin_col)
                #right_col_std = np.std(right_bin_col)
                #right_col_median = np.median(right_bin_col)
                #include_left_col =  [(elem <= sig_clipping_for_stats) for elem in np.abs(left_bin_col - np.poly1d(left_col_fit)(list(range(len(left_bin_col))))) / left_col_std]
                #include_right_col = [elem <= sig_clipping_for_stats for elem in np.abs(right_bin_col - np.poly1d(right_col_fit)(list(range(len(right_bin_col))))) / right_col_std]
                included_rows_in_fit = c.intersection(left_clipped_rows, right_clipped_rows)
                #include_row_in_fit = [(abs(left_bin_col[i] - left_col_median) / left_col_std) ** 2.0 + ((right_bin_col[i] - right_col_median) / right_col_std) ** 2.0 <= sig_clipping_for_stats ** 2.0 for i in range(len(left_bin_col))]
                #print ('included_rows_in_fit = ' + str(included_rows_in_fit))
                left_col_clipped = [left_bin_col[row] for row in included_rows_in_fit]
                right_col_clipped = [right_bin_col[row] for row in included_rows_in_fit]
                #left_of_right_fit = c.polyFitNSigClipping(right_col_clipped, left_col_clipped, amp_fit_order, sig_clipping_for_stats)
                mean_right_col_clipped = np.mean(right_col_clipped)
                mean_left_col_clipped = np.mean(left_col_clipped)
                left_of_right_fit = np.polyfit(np.array(right_col_clipped) - mean_right_col_clipped, np.array(left_col_clipped) - mean_left_col_clipped, amp_fit_order)
                left_of_right_fit = [left_of_right_fit[0], left_of_right_fit[1] + mean_left_col_clipped - left_of_right_fit[0] * mean_right_col_clipped]

                plt.scatter(right_bin_col, left_bin_col, c = 'r', marker = '.')
                plt.scatter(right_col_clipped, left_col_clipped, c = 'g', marker = '.')
                fit_plot = plt.plot(right_bin_col, np.poly1d(left_of_right_fit)(right_bin_col), c = 'k')[0]
                print ('left_of_right_fit = ' + str(left_of_right_fit))
                plt.title(file_to_analyze)
                plt.legend([fit_plot], ['fit params = ' + ','.join([str(c.round_to_n(elem, 3)) for elem in left_of_right_fit])] )
                #plt.draw()
                #plt.pause(2.0)
                #plt.close()
                plt.show()

                print ('[left_col_fit, right_col_fit,  ratio_fit, difference_fit] = '  +str([left_col_fit, right_col_fit,  ratio_fit, difference_fit]))
                self.analyzed_files = self.analyzed_files + [file_to_analyze]
                self.left_amp_fits[file_to_analyze] = left_col_fit
                self.right_amp_fits[file_to_analyze] = right_col_fit
                self.ratios_fits[file_to_analyze] = ratio_fit
                self.differences_fits[file_to_analyze] = difference_fit
                self.left_of_right_fits[file_to_analyze] = left_of_right_fit
            print ('analyzed_files ' + str(files_to_analyze))

        return 1




    def __init__(self, target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/PISCO/Feb_2020/ut200301/'):
        self.target_dir = target_dir
        self.shared_commands = spc.CommandHolder()
        self.filt_strs = self.shared_commands.getFilters()
        self.colors = ['g','r','pink','cyan']
        self.reducer = prc.PISCOReducer(target_dir = self.target_dir)
        self.analyzed_files = []
        self.amp_ratios = {}
        self.amp_differences = {}
        self.left_amp_fits = {}
        self.right_amp_fits = {}
        self.ratios_fits = {}
        self.differences_fits = {}
        self.left_of_right_fits = {}
        self.amp_sample_width = self.shared_commands.getAmpGainCorrectionWidth()
        self.binning = self.shared_commands.getBinning()
        self.precrop_amp_junctures = [self.shared_commands.getOverscanSections1x1()[det_num*2][0] // self.binning for det_num in range(len(self.filt_strs))]
        self.crop_sections = self.shared_commands.get1x1CropRegion()
        self.amp_junctures = [self.precrop_amp_junctures[det_num] - self.crop_sections[det_num][0][0] for det_num in range(len(self.precrop_amp_junctures)) ]

        print ('Hi. ')



if __name__ == '__main__':
    target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/PISCO/Feb_2020/ut200301/'
    #flat_list = ['Twiflat_' + str(num) + '.fits' for num in list(range(99, 107)) + list(range(109, 118)) + list(range(239, 247)) + list(range(248, 260))]
    shared_commands = spc.CommandHolder()
    filt_strs = shared_commands.getFilters()
    filt_strs = ['g']
    colors = ['g','r','pink','cyan']
    flat_lists = ['FLAT_' + filt_str + '.list' for filt_str in filt_strs]
    print ('flat_lists = ' + str(flat_lists))
    #flat_list = ['Twiflat_' + str(num) + '.fits' for num in list(range(101, 107))]
    flat_reducer = prc.PISCOReducer(target_dir = target_dir)
    y_lims = [0.98, 1.02]

    fig_size = [14, 7]

    amp_ratios = [[] for flat_list in flat_lists]
    amp_differences = [[] for flat_list in flat_lists]
    left_medians = [[] for flat_list in flat_lists]
    right_medians = [[] for flat_list in flat_lists]

    amp_sample_width = shared_commands.getAmpGainCorrectionWidth()
    binning = shared_commands.getBinning()
    precrop_amp_junctures = [shared_commands.getOverscanSections1x1()[det_num*2][0] // binning for det_num in range(len(filt_strs))]
    crop_sections = shared_commands.get1x1CropRegion()
    amp_junctures = [precrop_amp_junctures[det_num] - crop_sections[det_num][0][0] for det_num in range(len(precrop_amp_junctures)) ]
    print ('amp_junctures = ' + str(amp_junctures))
    flat_file_nums = [[] for flat_list in flat_lists]

    for band_num in range(len(filt_strs)):
        filt_str = filt_strs[band_num]
        flat_list = flat_lists[band_num]
        flat_files = np.loadtxt(target_dir + flat_list, dtype='str')
        print ('flat_files = ' + str(flat_files))
        flat_file_nums[band_num] = [int(flat_file[len('Twiflat_'):-len('.fits')]) for flat_file in flat_files]
        print ('flat_file_nums = ' + str(flat_file_nums))
        amp_juncture = amp_junctures[band_num]
        for flat_num in range(len(flat_files)):
            flat_file = flat_files[flat_num]
            print ('flat_file = ' + str(flat_file))
            processed_data_file = flat_reducer.fullReduceImage(flat_file, bias_correct = 1, flat_correct = 1, crop_image = 1, gain_correct = 0)[band_num]
            processed_data_array,processed_data_header = c.readInDataFromFitsFile(processed_data_file, target_dir)
            left_col = processed_data_array[:, (amp_juncture- amp_sample_width):amp_juncture]
            left_bin_col = np.sum(left_col, axis = 1)
            right_col = processed_data_array[:, amp_juncture:amp_juncture + amp_sample_width]
            right_bin_col = np.sum(right_col, axis = 1)
            #ratio = left_bin_col / right_bin_col
            #difference = left_bin_col - right_bin_col
            #median_ratio = np.median(ratio)
            #median_difference = np.median(difference)
            median_left_col = np.median(left_col)
            median_right_col = np.median(right_col)
            difference_of_median = median_left_col - median_right_col
            ratio_of_median = median_left_col / median_right_col
            amp_ratios[band_num] = amp_ratios[band_num] + [ratio_of_median]
            amp_differences[band_num]  = amp_differences[band_num] + [difference_of_median]
            left_medians[band_num] = left_medians[band_num] + [median_left_col]
            right_medians[band_num] = right_medians[band_num] + [median_right_col]
            print ('[ratio_of_median, difference_of_median,  median_left_col, median_right_col] = '  +str([ratio_of_median, difference_of_median,  median_left_col, median_right_col]))

    print ('[amp_ratios[0], amp_differences[0], left_medians[0], right_medians[0]] = ' + str([amp_ratios[0], amp_differences[0], left_medians[0], right_medians[0]] ))
    f, axarr = plt.subplots(3,len(filt_strs), squeeze = False, figsize = fig_size)
    for band_num in range(len(filt_strs)):
        single_band_ratios = amp_ratios[band_num]
        single_band_differences = amp_differences[band_num]
        single_band_left_medians = left_medians[band_num]
        single_band_right_medians = right_medians[band_num]

        single_band_ratios_from_differences = np.array(single_band_left_medians) / (np.array(single_band_left_medians) - np.array(single_band_differences))
        single_band_differences_from_ratios = np.array(single_band_left_medians) - (np.array(single_band_left_medians) / np.array(single_band_ratios))

        filt_str = filt_strs[band_num]
        flat_file_num = flat_file_nums[band_num]

        axarr[0, band_num].scatter(list(range(len(flat_file_num))), single_band_ratios, c = colors[band_num])
        axarr[0, band_num].scatter(list(range(len(flat_file_num))), single_band_ratios_from_differences, c = colors[band_num], marker = 'x')
        axarr[0, band_num].set_ylabel('amp ratio')
        xticks = [elem for elem in list(range(len(flat_file_num))) if elem % 4 == 0]
        axarr[0, band_num].set_xticks(xticks)
        axarr[0, band_num].set_xticklabels([flat_file_num[tick] for tick in xticks])
        axarr[0, band_num].set_ylim(y_lims)
        axarr[1, band_num].scatter(list(range(len(flat_file_num))), single_band_differences, c = colors[band_num])
        axarr[1, band_num].scatter(list(range(len(flat_file_num))), single_band_differences_from_ratios, c = colors[band_num], marker = 'x')
        axarr[1, band_num].set_ylabel('amp diff.')
        axarr[1, band_num].set_xticks(xticks)
        axarr[1, band_num].set_xticklabels([flat_file_num[tick] for tick in xticks])
        #axarr[1, band_num].set_ylim(y_lims)
        axarr[2, band_num].scatter(list(range(len(flat_file_num))), single_band_right_medians, c = colors[band_num])
        axarr[2, band_num].set_ylabel('Right amp med (ADU)')
        axarr[2, band_num].set_xticks(xticks)
        axarr[2, band_num].set_xticklabels([flat_file_num[tick] for tick in xticks])
        axarr[0, band_num].set_title(filt_str + '-band')
        axarr[2, band_num].set_xlabel('Flat image number')
        #axarr[2, band_num].set_ylim(y_lims)
    plt.tight_layout()
    plt.show()
