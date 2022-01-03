import SharedPISCOCommands as spc
import cantrips as c


class CatalogueObject:

    def generateFileForAstrometry(self, text_pos_file_suffix = None, fits_pos_file_suffix = None, target_dir = None, x_key_str = 'xcentroid', y_key_str = 'xcentroid', brightness_key_str = 'flux', object_number_key_str = 'obj_number',
                                  use_full_dict = 1, include_obj_nums = 0, include_fluxes = 0, save_as_text = 1 ):

        if text_pos_file_suffix == None:
            text_pos_file_suffix = self.shared_commands.getXYPositionTextFileSuffix()
        if fits_pos_file_suffix == None:
            fits_pos_file_suffix = self.shared_commands.getXYPositionFitsFileSuffix()
        if target_dir == None:
            target_dir = self.target_dir
        print ('self.cat_file = ' + str(self.cat_file))
        text_file_name = self.cat_file[0:-4] + text_pos_file_suffix #+ '.txt'\
        fits_file_name = self.cat_file[0:-4] + fits_pos_file_suffix

        if use_full_dict:
            dict_to_use = self.full_dict
        else:
            dict_to_use = self.trimmed_dict
        obj_nums = dict_to_use[object_number_key_str]
        xs = dict_to_use[x_key_str]
        ys = dict_to_use[y_key_str]
        brightnesses = dict_to_use[brightness_key_str]

        sorted_brightnesses, sorted_xs, sorted_ys, sorted_nums = c.safeSortOneListByAnother(brightnesses, [brightnesses, xs, ys, obj_nums])
        sorted_xs.reverse()
        sorted_ys.reverse()
        sorted_nums.reverse()
        sorted_brightnesses.reverse()

        #Do we want to save the file as a text file in columns or as a fits binary table?
        if save_as_text:
            if include_obj_nums:
                if include_fluxes:
                    lines = [ [sorted_nums[i], sorted_xs[i], sorted_ys[i], sorted_brightnesses[i]] for i in range(len(sorted_brightnesses)) ]
                else:
                    lines = [ [sorted_nums[i], sorted_xs[i], sorted_ys[i]] for i in range(len(sorted_brightnesses)) ]
            else:
                if include_fluxes:
                    lines = [ [sorted_xs[i], sorted_ys[i], sorted_brightnesses[i]] for i in range(len(sorted_brightnesses)) ]
                else:
                    lines = [ [sorted_xs[i], sorted_ys[i]] for i in range(len(sorted_brightnesses)) ]
            with open(target_dir + text_file_name, 'w') as f:
                if self.verbose: print ('Saving object positions to ' + str(target_dir + file_name))
                for line in lines:
                    f.write(' '.join([str(elem) for elem in line]) + '\n')
            return text_file_name
        else:
            if include_obj_nums:
                if include_fluxes:
                    col_names = ['OBJNum', 'X', 'Y', 'FLUX']
                    data = [sorted_nums, sorted_xs, sorted_ys, sorted_brightnesses]
                    formats = ['I', 'f4', 'f4', 'f4']
                else:
                    col_names = ['OBJNum', 'X', 'Y']
                    data = [sorted_nums, sorted_xs, sorted_ys]
                    formats = ['I', 'f4', 'f4']
            else:
                if include_fluxes:
                    col_names = ['X', 'Y', 'FLUX']
                    data = [sorted_xs, sorted_ys, sorted_brightnesses]
                    formats = ['f4', 'f4', 'f4']
                else:
                    col_names = ['X', 'Y']
                    data = [sorted_xs, sorted_ys]
                    formats = ['f4', 'f4']
            print ('col_names = ' + str(col_names))
            c.saveDataToFitsFile(data, fits_file_name, target_dir, col_formats = formats, col_names = col_names, data_type = 'table' )
            return fits_file_name


    #Must be updated
    def generateRegionFile(self, file_name = 'test.reg', region_type = 'ellipse', x_key_str = 'X_IMAGE', y_key_str = 'Y_IMAGE', color = 'green',
                                 A_key_str = 'A_IMAGE', B_key_str = 'B_IMAGE', theta_key_str = 'THETA_IMAGE', fwhm_str = 'FWHM_IMAGE',
                                 elongation_key_str = 'ELONGATION'):
        file_lines = ['# Region file format: DS9 version 4.1',
                      'global color='+ color  + ' dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                      'image']
        if region_type.lower() in ['ellipse', 'el', 'ell']:
            region_start = 'ellipse'
            #region_keys = [x_key_str, y_key_str, A_key_str, B_key_str, theta_key_str]
            region_keys = [x_key_str, y_key_str, fwhm_str, elongation_key_str, theta_key_str]
            for i in range(len(self.trimmed_dict[self.sex_keys[0]])):
                file_lines = file_lines + [region_start + '(' + ','.join([str(self.trimmed_dict[x_key_str][i]), str(self.trimmed_dict[y_key_str][i]),
                                                                          str(self.trimmed_dict[fwhm_str][i] * np.sqrt(self.trimmed_dict[elongation_key_str][i]) * 0.5),
                                                                          str(self.trimmed_dict[fwhm_str][i] / np.sqrt(self.trimmed_dict[elongation_key_str][i]) * 0.5),
                                                                          str(self.trimmed_dict[theta_key_str][i]) ])  + ')']
        elif region_type.lower() in ['circle', 'circ']:
            region_start = 'circle'
            region_keys = [x_key_str, y_key_str, fwhm_str]
            for i in range(len(self.trimmed_dict[self.sex_keys[0]])):
                file_lines = file_lines + [region_start + '(' + ','.join([str(self.trimmed_dict[x_key_str][i]), str(self.trimmed_dict[y_key_str][i]),
                                                                          str(self.trimmed_dict[fwhm_str][i] * 0.5) ])  + ')']
        elif region_type.lower() in ['fwhm', 'fwhm_measured_stars']:
            region_start = 'circle'
            region_keys = [x_key_str, y_key_str, fwhm_str]
            for i in range(len(self.fwhm_dict[x_key_str])):
                file_lines = file_lines + [region_start + '(' + ','.join([str(self.fwhm_dict[x_key_str][i]), str(self.fwhm_dict[y_key_str][i]),
                                                                          str(self.fwhm_dict[fwhm_str][i] * 0.5) ])  + ')']

        with open(file_name, 'w') as f:
            for line in file_lines:
                f.write("%s\n" % line)
        return 1

    def readInCatFile(self, ):
        catalogue_file_start_indicator_str = self.shared_commands.getIndicatorOfHeaderInCatalogue() # '#'
        cat_lines = c.readInRowsToList(self.cat_file, file_dir = self.target_dir, verbose = self.verbose)
        self.opening_lines = []
        property_dict = {}
        outputs = []
        for line in cat_lines:
            if line[0] == catalogue_file_start_indicator_str:
                self.opening_lines = self.opening_lines + [line]
                property_dict[line[2]] = []
            else:
                #if len(outputs) == 0:  outputs = [[] for elem in range(len(list(property_dict.keys())))]
                for j in range(len(property_dict.keys())):
                    key = list(property_dict.keys())[j]
                    try:
                        elem = float(line[j])
                    except(ValueError):
                        elem = line[j]
                    property_dict[key] = property_dict[key] + [elem]
                    #if j == len(outputs) - 1: outputs[j][-1] = float(outputs[j][-1][0:-1])
                    #else: outputs[j][-1] = float(outputs[j][-1])
        self.property_dict = property_dict
        self.full_dict = property_dict.copy()

        return 1

    def __init__(self, cat_file, target_dir, verbose = 0, ):
        self.verbose = verbose
        self.target_dir = target_dir
        self.original_cat_file = cat_file
        self.cat_file = cat_file

        self.shared_commands = spc.CommandHolder()

        self.readInCatFile()
