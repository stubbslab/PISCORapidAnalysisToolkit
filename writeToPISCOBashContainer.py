import numpy as np
import sys
import bashVarContainer as bvc


if __name__=="__main__":

    command_line_args = sys.argv[1:]
    #print ('command_line_args = ' + str(command_line_args))
    container_file, field_root_name, data_dir, binning, overscan, do_bias, do_flat, do_sextraction, correct_amp_step, clean_extras, crc_corrected, n_crc_corrections, crc_prefix, processed_prefix, wcs_prefix, wcs_solution_suffix, wcs_positions_suffix, filters, default_center_filter, list_suffix, cat_file_suffix, unified_cat_file_suffix, master_cat_file_suffix, obj_positions_file_suffix, obj_positions_no_numbers_file_suffix, obj_positions_binary_table_suffix, api_key, processing_dir, astrometry_dir, colmerge_dir, running_seeing_data_file, running_seeing_img, match_tol_in_deg, wcs_comp_midfix, wcs_centering_suffix    = command_line_args
    var_retrieval_dict = {'container_file':container_file, #This file
                          'field_root_name':field_root_name,
                          'data_dir':data_dir,
                          'binning':binning,
                          'overscan':overscan,
                          'do_bias':do_bias,
                          'do_flat':do_flat,
                          'do_sextraction':do_sextraction,
                          'correct_amp_step':correct_amp_step,
                          'clean_extras':clean_extras,
                          'crc_corrected':crc_corrected,
                          'n_crc_corrections':n_crc_corrections,
                          'crc_prefix':crc_prefix,
                          'processed_prefix':processed_prefix,
                          'wcs_prefix':wcs_prefix,
                          'wcs_solution_suffix':wcs_solution_suffix,
                          'wcs_positions_suffix':wcs_positions_suffix,
                          'filters':filters,
                          'default_center_filter':default_center_filter,
                          'list_suffix':list_suffix,
                          'cat_file_suffix':cat_file_suffix,
                          'unified_cat_file_suffix':unified_cat_file_suffix,
                          'master_cat_file_suffix':master_cat_file_suffix,
                          'obj_positions_file_suffix':obj_positions_file_suffix,
                          'obj_positions_no_numbers_file_suffix':obj_positions_no_numbers_file_suffix,
                          'obj_positions_binary_table_suffix':obj_positions_binary_table_suffix,
                          'api_key':api_key,
                          'processing_dir':processing_dir,
                          'astrometry_dir':astrometry_dir,
                          'colmerge_dir':colmerge_dir,
                          'running_seeing_data_file':running_seeing_data_file,
                          'running_seeing_img':running_seeing_img,
                          'match_tol_in_deg':match_tol_in_deg,
                          'wcs_comp_midfix':wcs_comp_midfix,
                          'wcs_centering_suffix':wcs_centering_suffix
}

    piscoContainer = bvc.bashVarContainerObject(container_file, var_dict = var_retrieval_dict, container_dir = '', readInFromFile = False)
    piscoContainer.saveContainerToFile()
