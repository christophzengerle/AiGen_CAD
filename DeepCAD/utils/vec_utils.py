import sys

sys.path.append("..")
import numpy as np
from cadlib.macro import CMD_ARGS_MASK


# fix the predicted vectors
# remove the redundant commands and add the missing SOL - Tokens
def fix_pred_vecs(seq):
    if seq[:, 0][0] != 4:
        seq = np.insert(
            seq, 0, np.concatenate(([4], ~CMD_ARGS_MASK[4]), axis=0), axis=0
        )

    new_seq = seq.copy()
    state = False
    while state == False:
        cmds = seq[:, 0]
        state = True
        for i in range(1, len(cmds)):
            if cmds[i] == 5:
                if cmds[i - 1] == 5:
                    new_seq = np.delete(new_seq, (i), axis=0)
                    state = False

                if cmds[i - 1] == 4:
                    new_seq = np.delete(new_seq, (i - 1), axis=0)
                    state = False

                if i < len(cmds) - 1:
                    if cmds[i + 1] != 4 and cmds[i + 1] != 5:
                        new_seq = np.insert(
                            new_seq,
                            i + 1,
                            np.concatenate(([4], ~CMD_ARGS_MASK[4]), axis=0),
                            axis=0,
                        )
                        state = False

        seq = new_seq.copy()
    return np.array(seq)
