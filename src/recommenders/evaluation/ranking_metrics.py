def precision_at_k(relevancy_list, predicted_list, k):
    relevant_items = 0
    predicted_list = list(predicted_list)
    for item in predicted_list[:k]:
        if item in relevancy_list:
            relevant_items += 1
    return  relevant_items/k

def recall_at_k(relevancy_list, predicted_list, k):
    relevant_items = 0
    predicted_list = list(predicted_list)
    for item in predicted_list[:k]:
        if item in relevancy_list:
            relevant_items += 1
    return  relevant_items/len(relevancy_list)

def map():
    pass

def ncdg():
    pass

if __name__ == "__main__":
    pass