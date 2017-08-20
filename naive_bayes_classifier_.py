# dataset = [feature 1, feature 2, feature 3, feature 4, class]
dataset = [[1, 0, 1, 1, 0],
           [1, 1, 0, 0, 1],
           [1, 0, 2, 1, 1],
           [0, 1, 1, 1, 1],
           [0, 0, 0, 0, 0],
           [0, 1, 2, 1, 1],
           [0, 1, 2, 0, 0],
           [1, 1, 1, 1, 1]]

test_data = [1, 0, 0, 0]


# prior probabilities calculation:
def cal_prior_prob(dataset):
    size = len(dataset)
    classes = list(set(row[-1] for row in dataset))
    prior_prob_value = {}
    for cls in classes:
        prior_prob_value[cls] = [row[-1] for row in dataset].count(cls) / float(size)
    return prior_prob_value


# likelyhood probabilities calculation
def cal_likelyhood_prob(dataset, test_data, class_value):
    feature_prob = 1.0
    cls_count = [row[-1] for row in dataset].count(class_value)
    for feature in range(len(test_data)):
        feature_count = 0
        for row in dataset:
            if test_data[feature] == row[feature]:
                if row[-1] == class_value:
                    feature_count += 1
        feature_prob *= (feature_count / cls_count)
    return feature_prob


# making class prediction
def predict(naive_bayes_result):
    prob = 0.0
    predicted_class = 0
    answer = None
    for predicted_class in naive_bayes_result:
        if naive_bayes_result[predicted_class] > prob:
            prob = naive_bayes_result[predicted_class]
            answer = predicted_class
    return answer


# main function
def cal_naive_bayes(dataset, test_data):
    prior_prob_value = cal_prior_prob(dataset)
    result = {}
    for class_value in prior_prob_value:
        result[class_value] = (cal_likelyhood_prob(dataset, test_data, class_value)) * prior_prob_value[class_value]
    print(result)
    return predict(result)


print("The test data is going to be in class:", cal_naive_bayes(dataset, test_data))
