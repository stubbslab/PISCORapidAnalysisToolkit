#!/bin/bash
#This script initializes bash parameters that the subsequent scripts will use.
# They will identify the correct file containing these names
#  by the 'field_name' specified when this process is invoked, and so
#  that name should be kept the same throughout.

field_root_name=$1 #The name of the field or general grouping of images to be batch-processed
data_dir=$2 #The directory where the data is stored, including the trailing '/'
binning=${3:-1} #[OPTIONAL] The binning (1 or 2).  Default is 1.
overscan=${4:-1} #[OPTIONAL] Should we overscan correct the files during processing (should generally be 1)
do_bias=${5:-1} #[OPTIONAL] Should we bias correct the files during processing (should generally be 1)
do_flat=${6:-1} #[OPTIONAL] Should we flat correct the files during processing (should generally be 1)
do_sextraction=${7:-1} #[OPTIONAL] Should we use source extractor to look for objects? (should generally be 1)
correct_amp_step=${8:-1} #[OPTIONAL] Should we try to correct the gain step between adjacent amplifiers (should generally be 1)
clean_extras=${9:-0} #[OPTIONAL] Should the process clean non-essential files as it goes (crc solutions, column files) (0=NO, 1=YES).  Default 0.
crc_corrected=${10:-0} #[OPTIONAL] Should a cosmic ray correction be applied (0=NO, 1=YES).  Default 0
n_crc_corrections=${11:-3} #[OPTIONAL] How many crc_corrections should be done.  Only relavent if crc_corrected is 1.  Default 3.

#The following are other parameters that the PISCO RAT requires...
crc_prefix='crc_'
processed_prefix='proc_'
wcs_prefix='wcs_'
wcs_solution_suffix='_wcs_solution.fits'
wcs_positions_suffix='_wcs.txt'
#declare -a filters=('g' 'r' 'i' 'z')
declare -a filters=('g' 'r' 'i' 'z')
filters_str=''
for i in "${filters[@]}"; do filters_str=$filters_str"$i"","; done
filters_str="${filters_str%?}"
#filters_str=$filters_str
default_center_filter='r'
list_suffix='.list'
cat_file_suffix='.cat'
unified_cat_file_suffix='.ucat'
master_cat_file_suffix='.mcat'
obj_positions_file_suffix='_positions.txt'
obj_positions_no_numbers_file_suffix='_positions_no_numbers.txt'
obj_positions_binary_table_suffix='_positions_and_fluxes.xyls'
running_seeing_data_file="seeing.csv"
running_seeing_img="seeing.png"
wcs_centering_suffix="_centered_wcs"
wcs_comp_midfix="_wcs_VS_"
match_tol_in_deg=0.001  #colmerge match tolerance, in degrees
#...and the following should be changed by a new user:
api_key='htmlavmvxjouwzzn' #update with user's API key
processing_dir='/Users/sashabrownsberger/Desktop/PISCORapidAnalysisToolkit/' #update to point to unpacked RAT directory
#(end list of variables that a new user must change)
astrometry_dir="$processing_dir"'astrometry.net/net/client'
colmerge_dir="$processing_dir"'colmerge2/'


container_file=$field_root_name".bcf" # _PISCOProcessParams.txt"
#echo Here are the values to put:
#echo $container_file
#echo $field_root_name
#echo $data_dir
#echo $binning
#echo $clean_extras
#echo $crc_corrected
#echo $n_crc_corrections
#echo $crc_prefix
#echo $processed_prefix
#echo $wcs_prefix
#echo $wcs_solution_suffix
#echo $filters_str
#echo $match_tol_in_deg
#echo $default_center_filter
#echo $list_suffix
#echo $cat_file_suffix
#echo $unified_cat_file_suffix
#echo $master_cat_file_suffix
#echo $obj_positions_file_suffix
#echo $obj_positions_no_numbers_file_suffix
#echo $api_key
#echo $processing_dir
#echo $astrometry_dir
#echo $colmerge_dir
#echo ...and that is all of them.

python writeToPISCOBashContainer.py $container_file $field_root_name $data_dir $binning $overscan $do_bias $do_flat $do_sextraction $correct_amp_step $clean_extras $crc_corrected $n_crc_corrections $crc_prefix $processed_prefix $wcs_prefix $wcs_solution_suffix $wcs_positions_suffix $filters_str $default_center_filter $list_suffix $cat_file_suffix $unified_cat_file_suffix $master_cat_file_suffix $obj_positions_file_suffix $obj_positions_no_numbers_file_suffix $obj_positions_binary_table_suffix $api_key $processing_dir $astrometry_dir $colmerge_dir $running_seeing_data_file $running_seeing_img $match_tol_in_deg $wcs_comp_midfix $wcs_centering_suffix
