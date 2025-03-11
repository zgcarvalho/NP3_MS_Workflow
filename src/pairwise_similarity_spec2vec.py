# code from https://github.com/iomega/spec2vec_gnps_data_analysis/blob/master/notebooks/
# adapted notebooks 3 and 6
# Compute spec2vec similarities on mass spectra dataset
import os
import sys
import gensim
import time
import numpy as np
from matchms.importing import load_from_mgf
from matchms.filtering import normalize_intensities
from matchms.filtering import remove_peaks_around_precursor_mz
from matchms.filtering import add_precursor_mz
from matchms.filtering import select_by_relative_intensity
from matchms.similarity.spectrum_similarity_functions import find_matches
from spec2vec import Spec2Vec
from spec2vec import SpectrumDocument


# Pre-processing of spectrum before pairwise comparison - similar to how it is done in MSCluster and NP3
# normalize spectrum, remove peaks around precursor if trim_mz is T, remove very low intensity peaks
def pre_process_spectrum(s, trim_mz=False):
    s = normalize_intensities(s)
    s = add_precursor_mz(s)
    #
    if trim_mz is True:
        s = remove_peaks_around_precursor_mz(s, mz_tolerance=20)
    # remove very low intensity peaks
    if len(s.peaks) >= 10:
        s = select_by_relative_intensity(s, intensity_from=np.quantile(s.peaks.intensities, 0.75)*0.001)
    # return spectrum
    return s

def compute_matches_symmetric(spectra_list, bin_size):
    matches_matrix = []
    for i in range(len(spectra_list)):
        # set nan values to the lower triangular matrix
        matches_i = [np.nan] * i
        for j in range(i, len(spectra_list)):
            # in the diagonal return the number of peaks
            # in the upper triangular matrix return the number of matched peaks
            if i == j:
                matches_i.append(len(spectra_list[i].peaks))
            else:
                matches_i.append(len(find_matches(spectra_list[i].peaks.mz, spectra_list[j].peaks.mz,
                                                  tolerance=bin_size)))
        matches_matrix.append(matches_i)
    # return the matrix with the number of matched peaks as a np ndarray
    return numpy.array(matches_matrix)


# stop("Seven arguments must be supplied to create the pairwise similarity table:\n",
#      " 1 - Job name;\n",
#      " 2 - Path to the MGF file with the spectra to be compared pairwise or directory where the mgfs from the clustering job exists;\n",
#      " 3 - Path to the output folder to save the resulting similarity table;\n",
#      " 4 - The bin size to consider two fragmented peaks m/z's the same;\n",
#      " 5 - The spectra fragmented peaks scaling method: 0 - ln, 1 - no scale and x > 0 power scale;\n",
#      " 6 - A logical indicating if the spectra should be trimmed by the precursor mass;\n",
#      " 7 - The maximum difference allowed between precursor m/zs to search for shifted m/z fragments in the cosine computation (max_shift);\n",
#      " 8 - (optional) The number of cores to use for parallel processing. At least 2 are needed for parallellization. If 1 disable parallel processing (default).\n",
#      call . =FALSE)

# using model spec2vec_UniqueInchikeys_ratio05_filtered_iter_50
# scale_factor is fixed to be power of 0.5
def compute_pairwise_similarity_spec2vec(data_name, path_mgf, output_path, bin_size, trim_mz):
    # get script directory to set spec2vec model path
    # models from https://zenodo.org/records/3978054
    src_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(src_dir, "spec2vec_models", "spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model")

    # load mgf with consensus spectra
    spectra = list(load_from_mgf(path_mgf))

    print("number of spectra:", len(spectra))
    # pre process spectra
    # apply pre processing steps to the data
    spectra_preprocessed = [pre_process_spectrum(s, trim_mz) for s in spectra]

    # convert spectra to documents in spec2vec format
    tstart = time.time()
    spectra_documents = [SpectrumDocument(s, n_decimals=2) for s in spectra_preprocessed]
    tend = time.time()
    print(f"Time to create {len(spectra_documents)} documents: {tend - tstart} s.")

    # load spec2vec model
    model = gensim.models.Word2Vec.load(model_file)

    # compute pairwise comparison using spec2vec
    # Actual score calculation
    # Using Spec2Vec with intensity_weighting_power=0.5.
    # Calculate matrix of all-vs-all similarity scores.
    spec2vec_similarity = Spec2Vec(model, intensity_weighting_power=0.5, allowed_missing_percentage=50)
    tstart = time.time()
    similarity_matrix = np.round(spec2vec_similarity.matrix(spectra_documents, spectra_documents, is_symmetric=True),3)
    tend = time.time()
    # remove lower triangular matrix values and reset diagonal with 1.0
    similarity_matrix[np.tril_indices_from(similarity_matrix)] = np.nan
    np.fill_diagonal(similarity_matrix, 1.0)

    print(f"Calculated {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} scores in {tend - tstart} s.")
    print(f"Calculated {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} scores in {(tend - tstart) / 60} min.")

    # also compute matched peaks
    # the number of peaks is directly influenced by the filtering method, this result will be different from the R code
    tstart = time.time()
    matches_matrix = compute_matches_symmetric(spectra_preprocessed, bin_size)
    tend = time.time()
    print(f"Calculated {matches_matrix.shape[0]}x{matches_matrix.shape[1]} scores in {tend - tstart} s.")
    print(f"Calculated {matches_matrix.shape[0]}x{matches_matrix.shape[1]} scores in {(tend - tstart) / 60} min.")

    # save similarity table in the right format
    scans_number = [s.get("scans") for s in spectra_preprocessed]
    header_index = "parameters sim pairwise - spec2vec - scale_factor:0.5;bin_size:"+str(bin_size)+";trim_mz:"+str(trim_mz)
    # Store similarity matrix
    similarity_matrix = pd.DataFrame(similarity_matrix, index=scans_number, columns=scans_number)
    similarity_matrix.to_csv(os.path.join(output_path, "similarity_table_"+data_name+"_spec2vec.csv"),
              sep=',', na_rep="", index_label=header_index)
    # store matches matrix
    matches_matrix = pd.DataFrame(matches_matrix, index=scans_number, columns=scans_number)
    matches_matrix.to_csv(os.path.join(output_path, "similarity_table_matches_"+data_name+"_spec2vec.csv"),
              sep=',', na_rep="", index_label=header_index)

    # TODO create main process to be called by the workflow if spec2vec is to be used