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

def make_cnn_dataset(utt_dict, frame_size=128, step_size=64):
    segment_set = []
    for frame_mat in utt_dict.values():
        frame_len = len(frame_mat)
        if frame_len < frame_size:
            continue
        
        for start_idx in range(0, frame_len-frame_size+1, step_size):
            segment = frame_mat[start_idx:start_idx+frame_size]
            segment_set.append(segment)
        segment = frame_mat[frame_len-frame_size:]
        segment_set.append(segment)
    return segment_set

def make_spk_cnn_set(utt_dict, frame_size=128, step_size=64):
    """ only for VAE """
    cnn_dict = dict()
    for utt_id, frame_mat in utt_dict.items():
        # spk_id = utt_id[:3] # for WSJ
        spk_id = utt_id.split("_")[0] # for TIMIT
            
        frame_len = len(frame_mat)
        # if total length is smaller than pre-defined frame size
        if frame_len < frame_size:
            continue
        
        # make segment
        segment_set = cnn_dict.get(spk_id, [])
        for start_idx in range(0, frame_len-frame_size+1, step_size):
            segment = frame_mat[start_idx:start_idx+frame_size]
            segment_set.append(segment) # tuple
        cnn_dict[spk_id] = segment_set
        
    return cnn_dict

def make_cnn_dataset_and_lab(utt_dict, lab_dict, frame_size=128, step_size=64):
    segment_set = []
    for utt_id, feat_mat in utt_dict.items():
        lab = lab_dict.get(utt_id, [])
        lab_len = len(lab)
        feat_len = len(feat_mat)
        if lab_len == 0:
            print("No label for", utt_id)
            continue
        if lab_len != feat_len:
            print("Label length is different in", utt_id)
            print("Feat :", feat_len,"/ Lab :", lab_len)
            continue
        
        if feat_len < frame_size:
            continue
        for start_idx in range(0, feat_len-frame_size+1, step_size):
            feat_segment = feat_mat[start_idx:start_idx+frame_size]
            lab_segment = lab[start_idx:start_idx+frame_size]
            segment = (feat_segment, lab_segment)
            segment_set.append(segment)
        feat_segment = feat_mat[feat_len-frame_size:]
        lab_segment = lab[feat_len-frame_size:]
        segment = (feat_segment, lab_segment)
        segment_set.append(segment)        
    return segment_set

def simulate_packet_loss(feat_mat, loss_per=0, minibatch=False):
    assert 0 <= loss_per < 100, "Loss rate should be set within range [0, 100)"
    feat_mat = np.array(feat_mat)
    if loss_per == 0:
        return feat_mat
    if not minibatch:
        feat_dim = len(feat_mat[0])
        total_frame_num = len(feat_mat)
        loss_frame_num = total_frame_num * loss_per // 100
        loss_index_list = np.random.choice(total_frame_num, loss_frame_num, replace=False)

        for loss_index in loss_index_list:
            feat_mat[loss_index] = np.zeros(feat_dim)
    else:
        batch_size=len(feat_mat)
        feat_dim = len(feat_mat[0][0])
        total_frame_num = len(feat_mat[0])
        loss_frame_num = total_frame_num * loss_per // 100
        for idx in range(batch_size):
            loss_index_list = np.random.choice(total_frame_num, loss_frame_num, replace=False)
            for loss_index in loss_index_list:
                feat_mat[idx][loss_index] = np.zeros(feat_dim)
    return feat_mat

