import numpy as np
from macros import CMD_ARGS_MASK

def fix_pred_vecs(seq):
    cmds = seq[:, 0]
    args = seq[:, 1:]
    if cmds[0] != 4:
        seq = np.insert(cmds, 0, 4, axis=0)
        args = np.insert(args, 0, CMD_ARGS_MASK[4], axis=0)
    if cmds[-1] != 5:
        seq = np.append(cmds, 5)
        args = np.append(args, CMD_ARGS_MASK[5], axis=0)
    
    state = True
    for i in range(1, len(cmds)):
        if cmds[i] == 0 or cmds[i] == 1 or cmds[i] == 2:
            if not state:
                for j in reversed(range(len(cmds[i]))):
                    if cmds[j] == 4:
                        break
                    elif cmds[j] == 5:
                        seq = np.insert(cmds, j, 4, axis=0)
                        args = np.insert(args, j, CMD_ARGS_MASK[4], axis=0)
                        break
        
        
        
        # Wenn die erste Zelle mit einem anderen Befehl als 4 beginnt, fügen Sie eine 4 am Anfang ein
        if seq[i][0] != 4:
            seq[i] = [4] + seq[i]

        # Wenn die letzte Zelle mit einem anderen Befehl als 5 endet, fügen Sie eine 5 am Ende ein
        if seq[i][-1] != 5:
            seq[i] = seq[i] + [5]

        # Durchlaufen Sie jede Zelle in der Zeile
        for j in range(len(seq[i])-1):
            # Wenn eine 4 direkt gefolgt von einer 5 ist, fügen Sie eine 0 zwischen sie ein
            if seq[i][j] == 4 and seq[i][j+1] == 5:
                seq[i] = seq[i][:j+1] + [0] + seq[i][j+1:]

            # Wenn eine 0, 1 oder 2 nicht von einer 4 und einer 5 eingeschlossen ist, umgeben Sie sie mit 4 und 5
            if seq[i][j] in [0, 1, 2] and (seq[i][j-1] != 4 or seq[i][j+1] != 5):
                seq[i] = seq[i][:j] + [4] + [seq[i][j]] + [5] + seq[i][j+1:]


    mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
    out_args[mask] = -1
    # Konvertieren Sie die Sequenz zurück in ein Numpy-Array
    seq = np.array(seq)

    return seq