import bashVarContainer as bvc
import cantrips as c
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

class CommandHolder:

    #assume images are ordered: g-left, g-right, r-left, r-right, i-left, i-right, z-left, z-right
    def writeSingleBandImages(self, single_amp_images, headers, image_root, save_dir = None, verbose = 0):
        if save_dir == None:
            save_dir = self.getSharedCalDir()
        filt_strs = self.getFilters()
        single_band_suffixes = self.getSingleBandSuffixes()
        image_extension = self.getImageExtension()

        for filt_str_num in range(len(filt_strs)):
            filt_str = filt_strs[filt_str_num]
            header = headers[filt_str_num * 2]
            header['FILTER'] = filt_str
            left_amp_image = single_amp_images[2 * filt_str_num]
            right_amp_image = single_amp_images[2 * filt_str_num + 1]
            left_shape = np.shape(left_amp_image)
            right_shape = np.shape(right_amp_image)
            image = np.zeros([max(left_shape[0], right_shape[0]), left_shape[1] + right_shape[1]])
            image[:,0:left_shape[1]] = left_amp_image
            image[:, left_shape[1]:left_shape[1] + right_shape[1]] = right_amp_image
            save_file_name = image_root + single_band_suffixes[filt_str_num] + image_extension
            c.saveDataToFitsFile(np.transpose(image), save_file_name, save_dir, header = header, verbose = verbose)

    def loadPiscoImage(self, image):
        imagelist = []
        hdulist_temp = fits.open(image)
        header = hdulist_temp[0].header
        for i in list(range(1,9)):
            image_data = hdulist_temp[i].data
            imagelist.append(image_data.astype(float))
        hdulist_temp.close()
        imagearray = np.array(imagelist)
        #print ('np.shape(imagearray) = ' + str(np.shape(imagearray)))
        return imagearray, header

    def piscoCrop(self, image_file, band_num,
                  target_dir = '', save_crop = 1):
        crop_regions_1x1 = self.get1x1CropRegion()
        crop_prefix = self.getCropPrefix()
        binning = self.getBinning()
        crop_regions = [ [ [axis_bound // binning for axis_bound in axis_bounds] for axis_bounds in crop_regino_band] for crop_region_band in crop_regions ]
        print ('crop_regions = ' + str(crop_regions))
        crop_region = crop_regions[band_num]
        data, header = c.readInDataFromFitsFile(image_file, target_dir)

        low_x, high_x, low_y, high_y = crop_region
        cropped = data[low_y, high_y, low_x, high_x]

        if save_crop:
            crop_header_keyword
            header[crop_header_keyword] = 'Cropped ' + str(crop_region) + ' from ' + str(list(np.shape(data)))
            crop_save_name = crop_prefix = image_file
            c.saveDataToFitsFile(cropped, crop_save_name, target_dir, header = header)

        return 0


    def PISCOStitch(self, raw_image_segments): #loads the result of loadimage(), crops and combined the 8 raw PISCO images into 4 combined images returned as a 4-image 3D numpy array
        images=[]
        bin_1x1_cut = self.getBinning1x1Cut()
        binning = self.getBinning()
        n_mosaics = self.getNMosaics()
        bin_cut = bin_1x1_cut // binning

        #print ('Cutting amplifiers...')
        for i in list(range(0, n_mosaics)):
            images.append(np.delete(raw_image_segments[i],np.s_[bin_cut:],1))
        images=np.asarray(images)
        #print ('(np.shape(images) ) = ' + str(np.shape(images) ))
        #print ('Orienting amplifiers...')
        images[1]=np.flip(images[1],1)
        images[2]=np.flip(images[2],1)
        images[4]=np.flip(images[4],0)
        images[5]=np.flip(np.flip(images[5],0),1)
        images[6]=np.flip(np.flip(images[6],0),1)
        images[7]=np.flip(images[7],0)
        #print ('Stitching cut and flipped amplifiers...')
        comimages=[]
        comimages.append(np.hstack((images[0],images[1])))
        comimages.append(np.hstack((images[3],images[2])))
        comimages.append(np.hstack((images[4],images[5])))
        comimages.append(np.hstack((images[7],images[6])))
        return np.asarray(comimages)

    def WriteStitchedImageToSingleBandFiles(self, images, headers, file_name_root, save_dir = None, verbose = 0):
        if save_dir == None:
            save_dir = self.getSharedCalDir()
        filt_strs = self.getFilters()
        suffixes = self.getSingleBandSuffixes()
        filter_keyword = self.getFilterHeaderKeyword()
        n_filters = len(filt_strs)
        file_names = []
        for filt_num in range(n_filters):
            data = images[filt_num ]
            header = headers[filt_num ]
            header[filter_keyword] = filt_strs[filt_num]
            suffix = suffixes[filt_num ]
            file_name = file_name_root + suffix
            file_names = file_names + [file_name]
            if verbose: print ('Writing to file ' + str(save_dir + file_name))
            c.saveDataToFitsFile(np.transpose(data), file_name, save_dir, header = header)
        return file_names

    def overscanCorrect(self, unsplit_image_file, verbose = 1, target_dir = '',
                        return_data = 0, save_data = 1, apply_overscan = 1,
                        img_already_split = 1, binning = 1):
        oscan_prefix = self.getOverscanPrefix()
        image_extension = self.getImageExtension()
        overscan_fit_order = self.getOverscanFitOrder()
        overscan_buffer = self.getOverscanBuffer()
        binning = self.getBinning()
        filt_strs = self.getFilters()
        full_overscan_sections_1x1 = self.getOverscanSections1x1()
        single_band_suffix_prefix = self.getSingleBandSuffixPrefix()
        n_mosaic_extensions = self.getNMosaics()
        image_extension = self.getImageExtension()
        split_img_suffixes = c.flattenListOfLists([single_band_suffix_prefix + filt_str + side_suffix for side_suffix in self.getLeftRightAmpSuffixes()] for filt_str in filt_strs)
        split_save_suffixes = c.flattenListOfLists([single_band_suffix_prefix + filt_str + side_suffix for side_suffix in self.getLeftRightAmpSuffixes()] for filt_str in filt_strs)

        if img_already_split:
            file_names = [unsplit_image_file[0:-len(image_extension)] + split_img_suffix + image_extension for split_img_suffix in split_img_suffixes]
            amplifiers_data = [[] for file_name in file_names]
            headers = [[] for file_name in file_names]
            for file_num in range(len(file_names)):
                amplifiers_data[file_num], headers[file_num] = c.readInDataFromFitsFile(file_names[file_num], target_dir)

        else:
            amplifiers_data, headers = self.loadPiscoImage(unpslit_image_file)
        if verbose: print ('Starting overscan correction of image ' + unsplit_image_file)
        full_overscan_sections = (np.array(full_overscan_sections_1x1) // binning).tolist()
        #print ('full_overscan_sections = ' + str(full_overscan_sections))
        for i in range(np.shape(amplifiers_data)[0]):
            #print ('Computing overscan for ' + str(i+1) + 'th amplifier of ' + str(np.shape(amplifiers_data)[0]) + '...')
            overscan_section = [full_overscan_sections[i][0] + overscan_buffer // binning,
                                full_overscan_sections[i][1]]
            overscan = amplifiers_data[i][:,overscan_section[0]:overscan_section[1]]
            xs = range(np.shape(amplifiers_data[i])[1])
            ys = range(np.shape(amplifiers_data[i])[0])
            x_mesh, y_mesh = np.meshgrid(xs, ys)
            average_overscan = np.mean(overscan, axis = 1)
            overscan_fit = np.poly1d( np.polyfit(ys, average_overscan, overscan_fit_order) )
            #plt.scatter(ys, average_overscan)
            #plt.plot(ys, overscan_fit(ys), c = 'r')
            #plt.show()

            if apply_overscan: amplifiers_data[i] = amplifiers_data[i] - overscan_fit(y_mesh)
        if save_data:
            save_files = [oscan_prefix + unsplit_image_file[0:-len(image_extension)] + split_save_suffixes[mosaic_num] + image_extension for mosaic_num in range(n_mosaic_extensions)]

            for i in range(len(save_files)):
                save_file = save_files[i]
                single_amplifier_data = amplifiers_data[i]
                header = headers[i]
                if verbose: print ('Saving data to file ' + str(target_dir + save_file))
                c.saveDataToFitsFile(np.transpose(single_amplifier_data), save_file, target_dir, header = header)
        if return_data:
            return amplifiers_data, headers
        else:
            return save_files

    def rawSplitImageFile(self, image_to_split, load_dir, save_dir = None, n_mosaic_extensions = None, verbose = 0, save_suffixes = None):
        image_extension = self.getImageExtension()
        single_band_suffix_prefix = self.getSingleBandSuffixPrefix()
        amp_id_keyword = self.getAmpHeaderKeyword()
        amp_suffixes = self.getLeftRightAmpSuffixes()
        filter_keyword = self.getFilterHeaderKeyword()
        filt_strs = self.getFilters()
        if save_dir == None:
            save_dir = load_dir
        if n_mosaic_extensions == None:
            n_mosaic_extensions = self.getNMosaics()
        if save_suffixes == None:
            save_suffixes = c.flattenListOfLists([ single_band_suffix_prefix + filt_str + side_suffix for side_suffix in amp_suffixes ] for filt_str in filt_strs)
        data_arrays, headers = c.readInDataFromFitsFile(image_to_split, load_dir, n_mosaic_image_extensions = n_mosaic_extensions, data_type = 'image')
        master_header = headers[0]

        save_files = [image_to_split[0:-len(image_extension)] + save_suffixes[mosaic_num] + image_extension for mosaic_num in range(n_mosaic_extensions)]
        for mosaic_num in range(n_mosaic_extensions):
            data = data_arrays[mosaic_num]
            header = master_header
            header[amp_id_keyword] = amp_suffixes[mosaic_num % 2]
            header[filter_keyword] = filt_strs[mosaic_num // 2]
            save_file = save_files[mosaic_num]
            c.saveDataToFitsFile(np.transpose(data), save_file, save_dir, header = header)

        return save_files

    def getPISCOVar(self, var_key):
        return self.varContainer.getVarFromDictPython(var_key)

    def getAmpCorrectColRange(self):
        return c.recursiveStrToListOfLists(self.getPISCOVar('amp_correct_col_range1x1'), elem_type_cast = int)

    def getAmpGainCorrectionWidth(self):
        return int(self.getPISCOVar('amp_gain_cor_bin_width'))

    def getAmpGainCorrectionFitSigClip(self):
        return int(self.getPISCOVar('amp_gain_cor_fit_sig_clip'))

    def getAmpGainCorrectionFitOrder(self):
        return int(self.getPISCOVar('amp_gain_correction_fit_order'))

    def getAmpGainPixelThreshold(self): 
        return int(self.getPISCOVar('amp_gain_correct_count_threshold'))

    def getGainCorrectionPrefix(self):
        return str(self.getPISCOVar('gain_correct_prefix'))

    def getFilterHeaderKeyword(self):
        return str(self.getPISCOVar('filter_header_keyword'))

    def getNormSectKeyword(self):
        return str(self.getPISCOVar('normalization_sect_keyword'))

    def getAmpFitColRangeHeaderKeyword(self):
        return str(self.getPISCOVar('gain_fit_col_range_keyword'))

    def getAmpFitHeaderKeyword(self):
        return str(self.getPISCOVar('amp_fit_keyword'))

    def get1x1CropRegion(self):
        return c.recursiveStrToListOfLists((self.getPISCOVar('crop_1x1')), elem_type_cast = int)

    def  getNCombinedHeaderKeyword(self):
        return str(self.getPISCOVar('n_combined_header_keyword'))

    def getFilters(self):
        return c.recursiveStrToListOfLists(self.getPISCOVar('filters'))

    def getMasterFlatLabel(self):
        return str(self.getPISCOVar('master_flat_root'))

    def getMasterBiasLabel(self):
        return str(self.getPISCOVar('master_bias_root'))

    def getSingleFilterFlatFiles(self):
        single_filt_strs = self.getFilters()
        list_suffix = self.getListSuffix()
        flat_list_name = self.getMasterFlatLabel()
        return [flat_list_name + single_filt_str + list_suffix for single_filt_str in single_filt_strs]

    def getPixelThresholds(self):
        return [int(threshold) for threshold in c.recursiveStrToListOfLists(self.getPISCOVar('pixel_thresholds'))]

    def getPixelScale(self):
        return float(self.getPISCOVar('pixel_scale'))

    def getAmplifierGains(self):
        return [float(gain) for gain in c.recursiveStrToListOfLists(self.getPISCOVar('gains'))]

    def getSuffix(self):
        return str(self.getPISCOVar('single_band_suffix_suffix') )

    def getSingleBandSuffixPrefix(self):
        return str(self.getPISCOVar('single_band_suffix_prefix'))

    def getSingleBandSuffixes(self):
        single_band_suffix_prefix = self.getSingleBandSuffixPrefix()
        filt_strs = self.getFilters()
        image_extension = self.getImageExtension()
        return [single_band_suffix_prefix + filt_str + image_extension for filt_str in filt_strs]

    def getLeftRightAmpSuffixes(self):
        return c.recursiveStrToListOfLists((self.getPISCOVar('left_right_suffix')))

    #When stitching, we do need to cut off the right overscan section, as it sits at the juncture
    # between left and right amp.
    #This is separate from the crop sections.
    def getBinning1x1Cut(self):
        return int(self.getPISCOVar('binning_1x1_cut'))

    def getFlatSingleAmpMeasureSections1x1(self):
         return c.recursiveStrToListOfLists((self.getPISCOVar('flat_single_amp_measure_sections_1x1')), elem_type_cast = int)

    def getFlatStitchedMeasureSections1x1(self):
         return c.recursiveStrToListOfLists((self.getPISCOVar('flat_stitched_measure_sections_1x1')), elem_type_cast = int)

    def getGoodFlatADUs(self):
        return c.recursiveStrToListOfLists((self.getPISCOVar('flat_good_adu_levels')), elem_type_cast = int)

    def getAmpHeaderKeyword(self):
         return self.getPISCOVar('amp_keyword')

    def getExpTimeKeyword(self):
        return self.getPISCOVar('exp_time_keyword')

    def getBiasSubKeyword(self):
        return self.getPISCOVar('bias_sub_keyword')

    def getFlatNormKeyword(self):
         return self.getPISCOVar('flat_norm_keyword')

    def getDateObsKeyword(self):
        return self.getPISCOVar('date_obs_keyword')

    def getExpStartKeyword(self):
        return self.getPISCOVar('crop_keyword')

    def getCropKeyword(self):
        return self.getPISCOVar('exp_start_keyword')

    def getObjectMaxFluxKeywordPrefix(self):
        return self.getPISCOVar('obj_peak_flux_keyword_prefix')

    def getStarPositionKeywordPrefix(self):
        return self.getPISCOVar('star_position_keyword_prefix')

    def getGoodObjectKeywordPrefix(self):
        return self.getPISCOVar('good_object_keyword_prefix')

    def getStarVsGalKeywordPrefix(self):
        return self.getPISCOVar('star_gal_keyword_prefix')

    def getKeywordBandSuffix(self):
        return self.getPISCOVar('mcat_keyword_band_suffix')

    def getExpEndKeyword(self):
        return self.getPISCOVar('exp_end_keyword')

    def getBiasXPartitions(self):
        return int(self.getPISCOVar('bias_combine_x_partitions'))

    def getBiasYPartitions(self):
        return int(self.getPISCOVar('bias_combine_y_partitions'))

    def getNMosaics(self):
        return int(self.getPISCOVar('n_mosaic_extensions'))

    def getBinning(self):
        return int(self.getPISCOVar('binning'))

    def getCropPrefix(self):
        return str(self.getPISCOVar('crop_prefix'))

    def getOverscanPrefix(self):
        return str(self.getPISCOVar('overscan_prefix'))

    def getNormalizationPrefix(self):
         return str(self.getPISCOVar('normalization_prefix'))

    def getBiasSubPrefix(self):
        return str(self.getPISCOVar('bias_correction_prefix'))

    def getFlatNormPrefix(self):
        return self.getPISCOVar('flat_correction_prefix')

    def getOverscanFitOrder(self):
        return int(self.getPISCOVar('overscan_fit_order'))

    def getOverscanBuffer(self):
        return int(self.getPISCOVar('overscan_buffer'))

    def getOverscanSections1x1(self):
        return c.recursiveStrToListOfLists((self.getPISCOVar('overscan_sections_1x1')), elem_type_cast = int)

    def getListSuffix(self):
        return str(self.getPISCOVar('list_suffix'))

    def getBiasList(self):
        bias_label = self.getMasterBiasLabel()
        list_suffix = self.getListSuffix()
        return bias_label + list_suffix

    def getFlatList(self):
        flat_label = self.getMasterFlatLabel()
        list_suffix = self.getListSuffix()
        return flat_label + list_suffix

    def getSingleBandFlatLists(self):
        single_band_suffix_prefix = self.getSingleBandSuffixPrefix()
        flat_label = self.getMasterFlatLabel()
        list_suffix = self.getListSuffix()
        filt_strs = self.getFilters()
        return [flat_label + single_band_suffix_prefix + filt_str + list_suffix for filt_str in filt_strs]

    def getFullFlatList(self):
        return str(self.getPISCOVar('full_flat_list'))

    def getMasterBiasRoot(self):
        return str(self.getPISCOVar('master_bias_root'))

    def getMasterFlatRoot(self):
        return str(self.getPISCOVar('master_flat_root'))

    def getSharedCalDir(self):
        base_dir = str(self.getPISCOVar('pisco_base_dir'))
        cal_dir = str(self.getPISCOVar('pisco_cal_dir'))
        return base_dir + cal_dir

    def getImageExtension(self):
        return str(self.getPISCOVar('image_extension'))

    def getCatalogueExtension(self):
        return str(self.getPISCOVar('catalogue_extension'))

    def getXYPositionTextFileSuffix(self):
        return str(self.getPISCOVar('xypos_test_suffix'))

    def getXYPositionFitsFileSuffix(self):
        return str(self.getPISCOVar('xypos_fits_suffix'))

    def getIndicatorOfHeaderInCatalogue(self):
        return str(self.getPISCOVar('catalogue_header_identifier'))

    def getRoughRAKeyword(self):
        return str(self.getPISCOVar('rough_ra_keyword'))

    def getRoughDecKeyword(self):
        return str(self.getPISCOVar('rough_dec_keyword'))

    def getAstrometryLowScale(self):
        return float(self.getPISCOVar('astrometry_solver_low_scale'))

    def getAstrometryHighScale(self):
        return float(self.getPISCOVar('astrometry_solver_high_scale'))

    def getAstrometryScaleUnits(self):
        return str(self.getPISCOVar('astrometry_scale_units') )

    def getWCSPrefix(self):
        return str(self.getPISCOVar('wcs_prefix'))

    def __init__(self, defaults_file = 'PISCODefaults.txt', defaults_dir = '/Users/sashabrownsberger/Documents/sashas_python_scripts/PISCORapidAnalysisToolkit_Updated/'):
        self.defaults_file = defaults_dir + defaults_file
        self.varContainer = bvc.bashVarContainerObject(container_file = defaults_file, container_dir = defaults_dir, readInFromFile = True)
        #print ('self.varContainer.var_dict = ' + str(self.varContainer.var_dict))
