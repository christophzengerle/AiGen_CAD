import numpy as np

ALL_COMMANDS = ["Line", "Arc", "Circle", "EOS", "SOL", "Ext"]
LINE_IDX = ALL_COMMANDS.index("Line")
ARC_IDX = ALL_COMMANDS.index("Arc")
CIRCLE_IDX = ALL_COMMANDS.index("Circle")
EOS_IDX = ALL_COMMANDS.index("EOS")
SOL_IDX = ALL_COMMANDS.index("SOL")
EXT_IDX = ALL_COMMANDS.index("Ext")

EXTRUDE_OPERATIONS = [
    "NewBodyFeatureOperation",
    "JoinFeatureOperation",
    "CutFeatureOperation",
    "IntersectFeatureOperation",
]
EXTENT_TYPE = [
    "OneSideFeatureExtentType",
    "SymmetricFeatureExtentType",
    "TwoSidesFeatureExtentType",
]

PAD_VAL = -1
N_ARGS_SKETCH = 5  # sketch parameters: x, y, alpha, f, r
N_ARGS_PLANE = 3  # sketch plane orientation: theta, phi, gamma
N_ARGS_TRANS = 4  # sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 4  # extrusion parameters: e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT

SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_ARGS)])
EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_ARGS)])

CMD_ARGS_MASK = np.array(
    [
        [1, 1, 0, 0, 0, *[0] * N_ARGS_EXT],  # line
        [1, 1, 1, 1, 0, *[0] * N_ARGS_EXT],  # arc
        [1, 1, 0, 0, 1, *[0] * N_ARGS_EXT],  # circle
        [0, 0, 0, 0, 0, *[0] * N_ARGS_EXT],  # EOS
        [0, 0, 0, 0, 0, *[0] * N_ARGS_EXT],  # SOL
        [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_EXT],
    ]
)  # Extrude

NORM_FACTOR = (
    0.75  # scale factor for normalization to prevent overflow during augmentation
)

MAX_N_EXT = 10  # maximum number of extrusion
MAX_N_LOOPS = 6  # maximum number of loops per sketch
MAX_N_CURVES = 15  # maximum number of curves per loop
MAX_TOTAL_LEN = 60  # maximum cad sequence length
ARGS_DIM = 256

ACC_TOLERANCE = 1


def trim_vec_EOS(out_vec):
    out_command = out_vec[:, 0]
    if out_command.tolist().count(EOS_IDX) == 0:
        return out_vec
    seq_len = out_command.tolist().index(EOS_IDX)
    out_vec = out_vec[:seq_len]
    out_vec = fix_pred_vecs(out_vec)
    return out_vec


# fix the predicted vectors
# remove the redundant commands and add the missing SOL - Tokens
def fix_pred_vecs(seq):
    if seq[:, 0][0] != 4:
        seq = np.insert(
            seq, 0, np.concatenate(([4], ~CMD_ARGS_MASK[4]), axis=0), axis=0
        )

    state = False
    while state == False:
        cmds = seq[:, 0]
        state = True
        for i in range(1, len(cmds)):
            if cmds[i] == 5:
                if cmds[i - 1] == 5:
                    seq = np.delete(seq, (i), axis=0)
                    state = False
                    break

                if i - 1 > 0:
                    if cmds[i - 1] == 4:
                        seq = np.delete(seq, (i - 1), axis=0)
                        state = False
                        break

                if i < len(cmds) - 1:
                    if cmds[i + 1] != 4 and cmds[i + 1] != 5:
                        seq = np.insert(
                            seq,
                            i + 1,
                            np.concatenate(([4], ~CMD_ARGS_MASK[4]), axis=0),
                            axis=0,
                        )
                        state = False
                        break
    return np.array(seq)
