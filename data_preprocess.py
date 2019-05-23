import numpy as np

def min_max_normalize(x):
    """
    Get 1D np array or list, return x - max(x) / max(x) - min(x)
    Range will be [0, 1]
    """
    x_min = np.min(x)
    x_max = np.max(x)
    res = (x - x_min) / (x_max - x_min)
    return res

def mean_std_normalize(x):
    """
    Get 1D np array or list, return x - mean(x) / std(x)
    Range will be [-1, 1]
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    res = (x - x_mean) / x_std
    return res

def sum_normalize(x):
    """
    Get 1D np array or list, return x / sum(x)
    Range will be [0, 1]
    """
    x_sum = np.sum(x)
    res = x / x_sum
    return res

def power_normalize(x):
    """
    Get 1D np array or list, return x / sigma(exp(x)^2)
    x - exp(2 * x_1){1 + exp(2*(x_2-x_1)) + exp(2*(x_3-x_1)) + ...}
    """
    x_max = np.max(x)
    max_power = np.exp(2*x_max)
    sum_power = np.sum(np.exp(2*(x-x_max)))
    power = max_power * sum_power

    res = x - power
    return res

def matrix_normalize(origin_mat, axis=None, fcn_type="mean"):
    """
    normalize numpy matrix along given axis
    """
    fcn_book = {
        "max": min_max_normalize,
        "mean": mean_std_normalize,
        "sum": sum_normalize,
        "power": power_normalize
    }

    assert fcn_type in fcn_book.keys() ,"Wrong normalization type"

    norm_fcn = fcn_book[fcn_type]
    if axis == -1:
        norm_mat = norm_fcn(origin_mat)
    elif axis == 0:
        norm_mat = np.array([norm_fcn(row) for row in origin_mat])
    elif axis == 1:
        norm_mat = np.array([norm_fcn(col) for col in origin_mat.T]).T
    else:
        norm_mat = origin_mat
    
    return norm_mat

def make_cnn_dataset(utt_dict, input_size=128, step_size=64):
    segment_set = []
    for frame_mat in utt_dict.values():
    
        frame_len = len(frame_mat)
        if frame_len < input_size:
            continue
        
        for start_idx in range(0, frame_len-input_size+1, step_size):
            segment = frame_mat[start_idx:start_idx+input_size]
            segment_set.append(segment)
    return segment_set

