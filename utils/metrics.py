import numpy as np

#Error-based Backward Transfer
def e_bwt(total_error_lists, T):
    # T is the number of sequential tasks, total_error_lists contain prediction results of all previous tasks
    tmp = 0
    for i in range(0, T-1):
        error_T_i = total_error_lists[-1][i]
        error_i_i = total_error_lists[i][-1]
        tmp += error_T_i-error_i_i
    bwt = tmp/(T-1)

    return bwt
