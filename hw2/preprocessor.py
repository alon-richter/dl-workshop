import dictionaries
import numpy
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis, ProtParamData

WINDOW = 11

# TODO: check about SS, slicing windows, RSA

def calculate_RSA(protein):
    analyzes_seq = ProteinAnalysis(protein.upper())
    res = [None, None, None]
    res.extend(analyzes_seq.protein_scale(window=7, param_dict=ProtParamData.em))
    res.extend([None, None, None])
    return res

# we define True: the amino acid is in epitope, False: the amino acid is not in epitope
def generate_labels(protein):
    res=[]
    for amino_acid in protein:
        if amino_acid.isupper():
            res.append(1)
        else:
            res.append(0)
    return res

def generate_dataset(protein):
    X = []
    matrix = generate_matrix(protein).to_numpy()
    for i in range (5, len(protein) - 5):
        X.append(matrix[i-5 : i+6])
    return X

#
#
# def separte(protein, offset):
#     window_size = WINDOW
#     subseq = []
#     while offset < len(protein):
#         if offset + window_size < len(protein):
#             subseq.append([protein[offset:offset + window_size]])
#         else:
#             subseq.append([protein[offset:offset+len(protein)]])
#     res = []
#
#     for seq in subseq:
#         res.append(generate_matrix(seq))
#     return res
#
#

def generate_matrix(protein):
    arr_vol = mapping(protein, "volume")
    arr_hyd = mapping(protein, "hydrophobicity")
    arr_pol = mapping(protein, "polarity")
    arr_rsa = mapping(protein, "RSA")
    arr_s = mapping(protein, "SS")
    arr_type = mapping(protein, "type")
    data = {"volume": arr_vol, "hydrophobicity": arr_hyd, "polarity": arr_pol, "RSA": arr_rsa, "ss":arr_s,
            "type": arr_type}
    df = pd.DataFrame(data)
    return df.astype(float)


def mapping(protein, feature_name):
    res = []
    if feature_name == "volume":
        for amino_acid in protein:
            res.append(dictionaries.AMINO_ACIDS_VOL_DICT[amino_acid.upper()])
        return res
    if feature_name == "hydrophobicity":
        for amino_acid in protein:
            res.append(dictionaries.AMINO_ACIDS_HYD_DICT[amino_acid.upper()])
        return res
    if feature_name == "polarity":
        for amino_acid in protein:
            res.append(dictionaries.AMINO_ACIDS_POL_DICT[amino_acid.upper()])
        return res
    if feature_name == "type":
        for amino_acid in protein:
            res.append(dictionaries.AMINO_ACIDS_TYPE_DICT[amino_acid.upper()])
        return res
    if feature_name == "RSA":
        for amino_acid in protein:
            res.append(dictionaries.AMINO_ACIDS_RSA_DICT[amino_acid.upper()])
        return res
    if feature_name == "SS":
        for amino_acid in protein:
            res.append(dictionaries.AMINO_ACIDS_SS_DICT[amino_acid.upper()])
        return res
