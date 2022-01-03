import numpy as np
import sys
import cantrips as c
import SharedPISCOCommands as spc
import os
import SashasAstronomyTools as atools
import cantrips as c
import CatalogueObjectClass as catObClass
import subprocess
import AstrometrySolverClass as asc

class PISCOAnalyzer:

    def defineCatFileFormattingParams(self, keyword_order = None, descriptions_dict = None, unit_str_dict = None, keyword_val_type = None):

        if keyword_order == None:
            keyword_order = {'obj_number':1,
                             'xcentroid':2,
                             'ycentroid':3,
                             'flux':4,
                             'peak':5,
                             'sharpness':6,
                             'roundness1':7,
                             'flux_e':8,
                             'flux_eErr':9,}

        if keyword_val_type == None:
            keyword_val_type = {'obj_number':int,
                             'xcentroid':float,
                             'ycentroid':float,
                             'flux':float,
                             'peak':float,
                             'sharpness':float,
                             'roundness1':float,
                             'flux_e':float,
                             'flux_eErr':float,}

        if descriptions_dict == None:
            descriptions_dict = descriptions_dict = {'obj_number': 'Running object number',
                                                     'xcentroid': 'x-centroid of the brightest pixel',
                                                     'ycentroid': 'y-centroid of the brightest pixel',
                                                     'flux':'Total flux without background subtraction',
                                                     'peak':'Peak pixel',
                                                     'sharpness':'Object sharpness',
                                                     'roundness1':'Object roundness based on symmetry',
                                                     'flux_e':'Total gain-corrected flux without background subtraction',
                                                     'flux_eErr':'Uncertainty in gain-corrected flux without background subtraction', }
        if unit_str_dict == None:
            unit_str_dict = unit_str_dict = {'obj_number': '',
                                             'xcentroid': '[pixel]',
                                             'ycentroid': '[pixel]',
                                             'flux':'[count]',
                                             'peak':'[count]',
                                             'sharpness':'',
                                             'roundness1':'',
                                             'flux_e':'[e-]',
                                             'flux_eErr':'[e-]'}

        self.keyword_order = keyword_order
        self.descriptions_dict = descriptions_dict
        self.unit_str_dict = unit_str_dict
        self.keyword_val_type = keyword_val_type

        return 1


    def writeDataToCatFile(self, image_file, target_dir, data_arrays_to_write, data_keywords_to_write = None, cat_file_to_write_to = None, data_save_types = None  ):
        if data_keywords_to_write == None:
            data_keywords_to_write = self.all_keywords
        if data_save_types == None:
            keyword_to_val_types = self.keyword_val_type
            data_save_types = [keyword_to_val_types[key] for key in data_keywords_to_write]
        image_file_extension = self.shared_commands.getImageExtension()
        cat_file_extension = self.shared_commands.getCatalogueExtension()
        ordered_keywords_to_write = c.safeSortOneListByAnother([self.keyword_order[keyword] for keyword in data_keywords_to_write], [data_keywords_to_write])[0]
        catalogue_header_id_str = self.shared_commands.getIndicatorOfHeaderInCatalogue()
        opening_rows = [' '.join([catalogue_header_id_str, str(i + 1), data_keywords_to_write[i], self.descriptions_dict[data_keywords_to_write[i]], self.unit_str_dict[data_keywords_to_write[i]]]) for i in range(len(ordered_keywords_to_write))]
        if cat_file_to_write_to == None:
            cat_file_to_write_to = image_file[0:-len(image_file_extension)] + cat_file_extension

        with open(target_dir + cat_file_to_write_to, 'w') as f:
            for opening_row in opening_rows:
                f.write(opening_row + '\n')
        c.saveListsToColumns(data_arrays_to_write, cat_file_to_write_to, target_dir, append = True, type_casts = data_save_types)

        return cat_file_to_write_to

    def creatCatalogueFileForImage(self, image_file, target_dir = None, load_dir = None, init_kernel_in_pix = None, sig_clipping_for_stats = None, star_find_n_sig_threshold = None, fit_radius_scaling = None, bg_radii_scalings = None, cat_file_to_write_to = None):
        if target_dir == None:
            target_dir = self.target_dir
        if load_dir == None:
            load_dir = self.target_dir
        if init_kernel_in_pix == None:
            init_kernel_in_pix = self.kernel_in_pix
        if sig_clipping_for_stats == None:
            sig_clipping_for_stats = self.sig_clipping_for_stats
        if star_find_n_sig_threshold == None:
            star_find_n_sig_threshold = self.star_find_n_sig_threshold
        if fit_radius_scaling == None:
            fit_radius_scaling = self.fit_radius_scaling
        if bg_radii_scalings == None:
            bg_radii_scalings = self.bg_radii_scalings
        filter_keyword = self.shared_commands.getFilterHeaderKeyword()
        filters = self.shared_commands.getFilters()
        gains = self.shared_commands.getAmplifierGains()
        gain_dict = {filters[band_num]:gains[band_num] for band_num in range(len(filters))}

        image_array, image_header = c.readInDataFromFitsFile(image_file, load_dir)

        image_filter = image_header[filter_keyword]

        image_data = atools.getDAOStarFinderStats(image_file, target_dir, init_kernel_in_pix, sig_clipping_for_stats = sig_clipping_for_stats, star_find_n_sig_threshold = star_find_n_sig_threshold, fit_radius_scaling = fit_radius_scaling, bg_radii_scalings = bg_radii_scalings)

        xs = image_data[self.x_keyword]
        ys = image_data[self.y_keyword]
        ADU_flux = image_data[self.flux_keyword] #Not background subtracted
        gain = gain_dict[image_filter]
        e_flux = [single_ADU_flux / gain for single_ADU_flux in ADU_flux]
        e_flux_err = np.sqrt(e_flux)
        peak_ADUs = image_data[self.peak_keyword]
        sharpness = image_data[self.sharpness_keyword]
        roundness = image_data[self.roundness_keyword]
        numbers = range(1, len(xs) + 1)

        cat_file = self.writeDataToCatFile(image_file, target_dir, [numbers, xs, ys, ADU_flux, e_flux, e_flux_err, peak_ADUs, sharpness, roundness],
                                           [self.numbers_keyword, self.x_keyword, self.y_keyword, self.flux_keyword, self.e_flux_keyword, self.e_flux_err_keyword, self.peak_keyword, self.sharpness_keyword, self.roundness_keyword],
                                           cat_file_to_write_to = cat_file_to_write_to)

        return cat_file


    def defineParamKeywords(self, x_keyword = None, y_keyword = None, flux_keyword = None, peak_keyword = None,
                                  sharpness_keyword = None, roundness_keyword = None, numbers_keyword = None,
                                  e_flux_keyword = None, e_flux_err_keyword = None, other_keywords = None):
        if x_keyword == None:
            x_keyword = 'xcentroid'
        if y_keyword == None:
            y_keyword = 'ycentroid'
        if flux_keyword == None:
            flux_keyword = 'flux'
        if peak_keyword == None:
            peak_keyword = 'peak'
        if sharpness_keyword == None:
            sharpness_keyword = 'sharpness'
        if roundness_keyword == None:
            roundness_keyword = 'roundness1'
        if numbers_keyword == None:
            numbers_keyword = 'obj_number'
        if e_flux_keyword == None:
            e_flux_keyword = 'flux_e'
        if e_flux_err_keyword == None:
            e_flux_err_keyword = 'flux_eErr'
        if other_keywords == None:
            other_keywords = []

        self.x_keyword = x_keyword
        self.y_keyword = y_keyword
        self.flux_keyword = flux_keyword
        self.peak_keyword = peak_keyword
        self.sharpness_keyword = sharpness_keyword
        self.roundness_keyword = roundness_keyword
        self.numbers_keyword = numbers_keyword
        self.e_flux_keyword = e_flux_keyword
        self.e_flux_err_keyword = e_flux_err_keyword
        self.other_keywords = other_keywords

        self.all_keywords = [self.x_keyword, self.y_keyword, self.flux_keyword, self.peak_keyword, self.sharpness_keyword,
                             self.roundness_keyword, self.numbers_keyword, self.e_flux_keyword, self.e_flux_err_keyword] + self.other_keywords

        return 1

    def generateSortedStellarPositionsCat(self, cat_file, target_dir = None, include_obj_nums = 0, include_fluxes = 1, verbose = None):
        if target_dir == None:
            target_dir = self.target_dir
        if verbose == None:
            verbose = verbose
        include_obj_nums = int(include_obj_nums)
        #print ('Here include_obj_nums = ' + str(include_obj_nums) )
        CatObject = catObClass.CatalogueObject(cat_file, target_dir, verbose = verbose)

        stellar_positions_text_file = CatObject.generateFileForAstrometry(x_key_str = self.x_keyword, y_key_str = self.y_keyword,
                                                                     brightness_key_str = self.flux_keyword, object_number_key_str = self.numbers_keyword,
                                                                     use_full_dict = 1, include_obj_nums = include_obj_nums, include_fluxes = include_fluxes, save_as_text = 1)
        stellar_positions_fits_file = CatObject.generateFileForAstrometry(x_key_str = self.x_keyword, y_key_str = self.y_keyword,
                                                                     brightness_key_str = self.flux_keyword, object_number_key_str = self.numbers_keyword,
                                                                     use_full_dict = 1, include_obj_nums = include_obj_nums, include_fluxes = include_fluxes, save_as_text = 0)

        return stellar_positions_text_file, stellar_positions_fits_file



    def computeWCSForImage(self, image_file, image_file_with_wcs = None, target_dir = None, cat_file = None, sorted_stellar_positions_file = None, rm_intermediate_files = None):
        astrometry_solver = asc.AstrometrySolver()

        image_file_extension = self.shared_commands.getImageExtension()
        cat_file_extension = self.shared_commands.getCatalogueExtension()
        if target_dir == None:
            target_dir = self.target_dir
        if rm_intermediate_files == None:
            rm_intermediate_files = self.rm_intermediate_files
        if image_file_with_wcs == None:
            wcs_prefix = self.shared_commands.getWCSPrefix()
            image_file_with_wcs = wcs_prefix + image_file
        if cat_file == None:
            cat_file = image_file[0:-len(image_file_extension)] + cat_file_extension
        print ('[image_file, target_dir] = ' + str([image_file, target_dir]))
        image_data, image_header = c.readInDataFromFitsFile(image_file, target_dir)
        height, width = np.shape(image_data)
        if sorted_stellar_positions_file == None:
            if not(os.path.exists(self.target_dir + cat_file)):
                print ('Target catalogue file ' + cat_file + ' does not yet exist in directory ' + target_dir + '.  We will make it...')
                cat_file = self.creatCatalogueFileForImage(image_file, target_dir = target_dir)
            print ('cat_file = ' + str(cat_file))
            sorted_stellar_positions_txt_file, sorted_stellar_positions_fits_file = self.generateSortedStellarPositionsCat(cat_file, target_dir, include_obj_nums = 0 )
        rough_ra_keyword, rough_dec_keyword = [self.shared_commands.getRoughRAKeyword(), self.shared_commands.getRoughDecKeyword()]
        rough_ra_str, rough_dec_str = [image_header[rough_ra_keyword], image_header[rough_dec_keyword]]

        print ('[rough_ra_str, rough_dec_str] = ' + str([rough_ra_str, rough_dec_str]))

        astrometry_scale_low = self.shared_commands.getAstrometryLowScale()
        astrometry_scale_high = self.shared_commands.getAstrometryHighScale()
        astrometry_scale_units = self.shared_commands.getAstrometryScaleUnits()

        #Download healpix map and move to correct directory
        wcs_solution_file = astrometry_solver.solveField(sorted_stellar_positions_fits_file, target_dir, rough_ra_str, rough_dec_str, height, width, astrometry_scale_low, astrometry_scale_high, astrometry_scale_units)
        print ('[wcs_solution_file, image_file, image_file_with_wcs, target_dir] = ' + str([wcs_solution_file, image_file, image_file_with_wcs, target_dir] ) )

        astrometry_solver.updateFitsHeaderWithSolvedField(wcs_solution_file, image_file, image_file_with_wcs, target_dir, verbose = 0)
        if rm_intermediate_files:
            os.remove(target_dir + wcs_solution_file)
            os.remove(target_dir + sorted_stellar_positions_fits_file)
            os.remove(target_dir + sorted_stellar_positions_txt_file)

        return image_file_with_wcs


    def __init__(self, target_dir,
                 rm_intermediate_files = 1, kernel_in_pix = 10,
                 sig_clipping_for_stats = 3.0, star_find_n_sig_threshold = 4.0,
                 fit_radius_scaling = 5.0, bg_radii_scalings = [7.0, 8.0]):

        self.target_dir = target_dir
        self.rm_intermediate_files = rm_intermediate_files
        self.kernel_in_pix = int(kernel_in_pix)
        self.sig_clipping_for_stats = sig_clipping_for_stats
        self.star_find_n_sig_threshold = star_find_n_sig_threshold
        self.fit_radius_scaling = star_find_n_sig_threshold
        self.star_find_n_sig_threshold = star_find_n_sig_threshold
        self.fit_radius_scaling = star_find_n_sig_threshold
        self.bg_radii_scalings = bg_radii_scalings
        self.shared_commands = spc.CommandHolder()

        self.defineParamKeywords()
        self.defineCatFileFormattingParams()
