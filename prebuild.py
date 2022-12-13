import os
import pickle
import logging
from utils import get_trans_features,get_rrank_features
from utils import get_dataset,get_index,load_vec_txt
from confidence import tc_threshold,get_threshold,get_trans_confidence
from utils import get_path_index

logger = logging.getLogger()

def build_dataset(data_dir,result_dir,log_dir,w2v_k,max_p):
    # entity and relation files
    entity2id_file = os.path.join(data_dir,"entity2id.txt")
    relation2id_file = os.path.join(data_dir,"relation2id.txt")
    ent_vocab,ent_idx_word = get_index(entity2id_file)
    rel_vocab,rel_idx_word = get_index(relation2id_file)
    logger.info("entity vocab size:  (%d,%d)"%(len(ent_vocab),len(ent_idx_word)))
    logger.info("relation vocab size:  (%d,%d)"%(len(rel_vocab),len(rel_idx_word)))

    # TransE vector files
    entity2vec_file = os.path.join(data_dir,"FB15K_TransE_Entity2Vec_100.txt")
    relation2vec_file = os.path.join(data_dir,"FB15K_TransE_Relation2Vec_100.txt")
    relvec_k,relation2vec = load_vec_txt(relation2vec_file,rel_vocab,k_num=w2v_k)
    entvec_k,entity2vec = load_vec_txt(entity2vec_file,ent_vocab,k_num=w2v_k)
    logger.info("entity vector size:( %d ), relation vector size:( %d )"%(len(entity2vec),len(relation2vec)))

    # train files and test files
    train_file = os.path.join(data_dir,"conf_train2id.txt")
    test_file = os.path.join(data_dir,"conf_test2id.txt")
    train_triple,train_confidence = get_dataset(train_file)
    logger.info("train triple size :(%d), train confidence size :(%d)"%(len(train_triple),len(train_confidence)))
    test_triple,test_confidence = get_dataset(test_file)
    logger.info("test triple size :(%d), test confidence size :(%d)"%(len(test_triple),len(test_confidence)))

    # get the predict results of the triples
    testfile_KGC_h_t = os.path.join(data_dir,"h_t.txt")
    testfile_KGC_hr_ = os.path.join(data_dir,"hr_.txt")
    testfile_KGC__rt = os.path.join(data_dir,"_rt.txt")
    test_triple_KGC_h_t, test_confidence_KGC_h_t = get_dataset(testfile_KGC_h_t)
    test_triple_KGC_hr_, test_confidence_KGC_hr_ = get_dataset(testfile_KGC_hr_)
    test_triple_KGC__rt, test_confidence_KGC__rt = get_dataset(testfile_KGC__rt)
    print("test_triple_KGC_h_t size :(%d), test_triple_KGC_h_t size :(%d), test_triple_KGC_h_t size :(%d)"
                %(len(test_triple_KGC_h_t),len(test_triple_KGC_hr_),len(test_triple_KGC__rt)))

    tcthreshold_dict = tc_threshold(train_triple,entity2vec,relation2vec)
    train_transE = get_trans_confidence(tcthreshold_dict,train_triple,entity2vec,relation2vec)
    test_transE = get_trans_confidence(tcthreshold_dict,test_triple,entity2vec,relation2vec)

    logger.info("train trainsE size:(%d), test trainsE size:(%d)"%(len(train_transE),len(test_transE)))

    test_transE_hr_ = get_trans_confidence(tcthreshold_dict, test_triple_KGC_hr_, entity2vec, relation2vec)
    test_transE_h_t = get_trans_confidence(tcthreshold_dict, test_triple_KGC_h_t, entity2vec, relation2vec)
    test_transE__rt = get_trans_confidence(tcthreshold_dict, test_triple_KGC__rt, entity2vec, relation2vec)
    
    logger.info("test transE hr_ size:(%d), test transE h_t size:(%d)， test transE _rt size:(%d)"%
                  (len(test_transE_hr_),len(test_transE_h_t),len(test_transE__rt)))
    
    entity_rank_path = os.path.join(result_dir,"ResourceRank_4")
    dict_features = get_trans_features(entity_rank_path)
    rrkthreshold_dict = {}


    train_rrank = get_rrank_features(dict_features,train_triple)
    test_rrank = get_rrank_features(dict_features,test_triple)

    logger.info("train rrank size :(%d) , test rrank size :(%d)"%(len(train_rrank),len(test_rrank)))

    test_rrank_KGC_h_t = get_rrank_features(dict_features, test_triple_KGC_h_t)
    test_rrank_KGC_hr_ = get_rrank_features(dict_features, test_triple_KGC_hr_)
    test_rrank_KGC__rt = get_rrank_features(dict_features, test_triple_KGC__rt)

    logger.info("test rrank KGC h_t size :(%d), test rrank KGC hr_ size:(%d), test rrank KGC _rt size :(%d)"%
                (len(test_rrank_KGC_h_t),len(test_rrank_KGC_hr_),len(test_rrank_KGC__rt)))
    
    path_file = os.path.join(result_dir,"Path_4")
    train_path_h, train_path_t, train_path_r = get_path_index(path_file, max_p, train_triple, 0)
    test_path_h, test_path_t, test_path_r = get_path_index(path_file, max_p, test_triple, 0)
    train_path2_h, train_path2_t, train_path2_r = get_path_index(path_file, max_p, train_triple, 1)
    test_path2_h, test_path2_t, test_path2_r = get_path_index(path_file, max_p, test_triple, 1)
    train_path3_h, train_path3_t, train_path3_r = get_path_index(path_file, max_p, train_triple, 2)
    test_path3_h, test_path3_t, test_path3_r = get_path_index(path_file, max_p, test_triple, 2)
    logger.info("train_path size:(%d),test_path size:(%d)"%(len(train_path_h),len(test_path_h)))
    
    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t = get_path_index(path_file, max_p, test_triple_KGC_h_t, 0)
    test_path_h_hr_, test_path_t_hr_, test_path_r_hr_ = get_path_index(path_file, max_p, test_triple_KGC_hr_, 0)
    test_path_h__rt, test_path_t__rt, test_path_r__rt = get_path_index(path_file, max_p, test_triple_KGC__rt, 0)

    logger.info("Dataset created !")
    save_file_name = os.path.join(log_dir,"data2_TransE.pkl")
    with open(save_file_name,mode="wb") as wfp:
        pickle.dump([ent_vocab, ent_idx_word, rel_vocab, rel_idx_word,
                 entity2vec, entvec_k,
                 relation2vec, relvec_k,
                 train_triple, train_confidence,
                 test_triple, test_confidence,
                    test_triple_KGC_h_t, test_confidence_KGC_h_t,
                    test_triple_KGC_hr_, test_confidence_KGC_hr_,
                    test_triple_KGC__rt, test_confidence_KGC__rt,
                 tcthreshold_dict, train_transE, test_transE,
                    test_transE_h_t,
                    test_transE_hr_,
                    test_transE__rt,
                 rrkthreshold_dict, train_rrank, test_rrank,
                    test_rrank_KGC_h_t,
                    test_rrank_KGC_hr_,
                    test_rrank_KGC__rt,
                 max_p,
                 train_path_h, train_path_t, train_path_r,
                 test_path_h, test_path_t, test_path_r,
                 train_path2_h, train_path2_t, train_path2_r,
                 test_path2_h, test_path2_t, test_path2_r,
                 train_path3_h, train_path3_t, train_path3_r,
                 test_path3_h, test_path3_t, test_path3_r,
                    test_path_h_KGC_h_t, test_path_t_KGC_h_t, test_path_r_KGC_h_t,
                    test_path_h_hr_, test_path_t_hr_, test_path_r_hr_,
                    test_path_h__rt, test_path_t__rt, test_path_r__rt], wfp)
    

