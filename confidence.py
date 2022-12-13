import math
import numpy as np

def tc_threshold(tcDevExamples,entity2vec,relation2vec):
    threshold_dict = {}
    trans_dict = {}

    for tri in tcDevExamples:
        triple_s = entity2vec[tri[0]] + relation2vec[tri[2]] - entity2vec[tri[1]]
        trans_v = np.linalg.norm(triple_s,ord=2)
        if tri[2] not in trans_dict:
            trans_dict[tri[2]] = [(trans_v,tri[3])]
        else:
            trans_dict[tri[2]].append((trans_v,tri[3]))
    for key in trans_dict:
        threshold_dict[key] = get_threshold(trans_dict[key])
    return threshold_dict
def get_threshold(rrank):
    distance_flaglist = rrank
    distance_flaglist = sorted(distance_flaglist,key=lambda sp:sp[0],reverse=False)

    threshold = distance_flaglist[0][0] - 0.01
    max_value = 0
    current_val = 0

    for k in range(1,len(distance_flaglist)):
        if distance_flaglist[k-1][1] == 1:
            current_val += 1
        else:
            current_val -= 1
        if current_val > max_value:
            threshold = (distance_flaglist[k][0]+distance_flaglist[k-1][0])/2.0
            max_value = current_val
    return threshold
def get_trans_confidence(threshold_dict,tcExamples,entity2vec,relation2vec):
    all_conf = 0.0
    confidence_dict = []

    right = 0.0

    for triple in tcExamples:
        if triple[2] in threshold_dict:
            threshold = threshold_dict[triple[2]]
        else:
            threshold = 0.0
        val_s = entity2vec[triple[0]] + relation2vec[triple[2]]-entity2vec[triple[1]]
        trans_v = np.linalg.norm(val_s,ord=2)
        f = 1.0/(1.0 + math.exp(-1.0*(threshold-trans_v)))
        f = (threshold - trans_v)

        confidence_dict.append(f)

        if trans_v <= threshold and triple[3] == 1:
            right += 1.0
            all_conf += f
        elif trans_v >threshold and triple[3] == -1:
            right += 1.0
        else:
            pass
    
    print("Trans Confidence accuracy %0.2f"%(right/len(tcExamples)))

    avg_conf = all_conf /float(len(tcExamples))
    print("Average confidence %0.2f , %0.2f "%(avg_conf,float(len(tcExamples))))

    return confidence_dict





