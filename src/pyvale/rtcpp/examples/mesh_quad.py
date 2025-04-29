# make sure to install the imagebenchmarks repo to the same env for this
import imagebenchmarks

# (case_ident,case_mesh,case_camera) = imagebenchmarks.load_benchmark_by_tag("case0_plate_lintri_1Mpx_1subsamp_nocrop_11776elems")

(case_ident,case_mesh,case_camera) = imagebenchmarks.load_benchmark_by_tag("case71_plate_quadquad_24Mpx_2subsamp_crop_135168elems")

case_mesh
sim_data = mh.ExodusReader(data_path).read_all_sim_data()