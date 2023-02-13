"""
Модуль с метриками для рекомендательных систем
"""
import numpy as np


def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1


def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)


def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()


def recall(recommended_list, bought_list):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall


def recall_at_k(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = np.array(recommended_list[:k])
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):    
    recommend_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommend_list, bought_list)
    money_recall = np.dot(flags, prices_recommended).sum() / prices_bought.sum() 
    return money_recall


def ap_k(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):        
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k            
    return sum_ / sum(flags)


def mrr_at_k(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(recommended_list, bought_list)

    # если среди рекомендаций не было ни одной покупки
    if (not np.any(flags)):
        return np.inf

    rank = np.argmax(flags) + 1 
    return 1/rank


def ndcg_at_k(recommended_list, bought_list, k=5):    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k] 
    flags = np.isin(recommended_list, bought_list)
    i = np.arange(1, k+1)
    discount = np.where(i<=2, i, np.log2(i))
    dcg = (flags / discount).sum() / k
    idcg = (np.ones(discount.shape) / discount).sum() / k
    return dcg/idcg