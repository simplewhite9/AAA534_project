
ep = 1e-10

def getCount(freq):
    count, total = freq[0], freq[1]
    return count / total if total != 0 else 0.0

def updateDatasetMetric(dataset, metric_logger, q_freq):
    if dataset == 'causalvid':
        metric_logger.update(n=q_freq['descriptive'][1]+ep, descriptive = getCount(q_freq['descriptive']))
        metric_logger.update(n=q_freq['explanatory'][1]+ep, explanatory = getCount(q_freq['explanatory']))
        metric_logger.update(n=q_freq['pred_A'][1]+ep, pred_A = getCount(q_freq['pred_A']))
        metric_logger.update(n=q_freq['pred_R'][1]+ep, pred_R = getCount(q_freq['pred_R']))
        metric_logger.update(n=q_freq['pred_AR'][1]+ep, pred_AR = getCount(q_freq['pred_AR']))
        metric_logger.update(n=q_freq['countf_A'][1]+ep, countf_A = getCount(q_freq['countf_A']))
        metric_logger.update(n=q_freq['countf_R'][1]+ep, countf_R = getCount(q_freq['countf_R']))
        metric_logger.update(n=q_freq['countf_AR'][1]+ep, countf_AR = getCount(q_freq['countf_AR']))