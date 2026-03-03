import math

def validate_data(data):
    if not data:
        raise ValueError("Data cannot be empty")


def mean(data):
    validate_data(data)
    return sum(data)/len(data)  #To find the mean of the given data



def Variance(data):
    mu = mean(data)
    return sum((x-mu)**2 for x in data)/len(data) #To find the variance of the given data


def std_dev(data):
    return math.sqrt(Variance(data))  #To find the standard_deviation of the given data


def maximum(data):
    max_val = data[0]
    for x in data:
       if x > max_val:
           max_val = x
    return max_val    #Find the maximum number in the given data

def minimum(data):
    min_val = data[0]
    for x in data:
        if x < min_val:
            min_val = x
    return min_val      #Find the minimum number in the given data


def custom_sum(data):
    total = 0
    for x in data:
        total += x
    return total      # To Find the total sum of the given data


def vec_add(v1,v2):
    return [v1[i]+v2[i] for i in range(len(v1))]   #To find the addition of the given vector


def vec_sub(v1,v2):
    return [v1[i]-v2[i] for i in range (len(v1))]    #To find the subtraction of the given vector

def Scalar_multiplication(scalar,vector):
    return [scalar*x for x in vector]     #To find the scalar mutliply


def z_scores(data):
    mu = mean(data)
    sigma = std_dev(data)
    return [(x - mu) / sigma for x in data]   #To find the z_score of the given data

def percentile(data,p):
    data = sorted(data)
    n = len(data)

  #calculate index
    index = (p/100) * (n-1)

    lower = int(index)
    upper = lower + 1

    if index == lower:
        return data[lower]
    
    fraction = index - lower
    return data[lower] + fraction * (data[upper] - data[lower])

def quartiles(data):
    Q1 = percentile(data, 25)
    Q2 = percentile(data, 50)
    Q3 = percentile(data, 75)
    
    return Q1, Q2, Q3

def iqr(data):
    Q1, Q2, Q3 = quartiles(data)
    return Q3 - Q1


def min_max_normalization(data):
    validate_data(data)
    min_val = minimum(data)
    max_val = maximum(data)

    if max_val == min_val:
        return [0 for _ in data]
    return [(x-min_val)/(max_val-min_val) for x in data]


# Classification Metrics


def confusion_matrix(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")

    TP = TN = FP = FN = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            TP += 1
        elif actual == 0 and predicted == 0:
            TN += 1
        elif actual == 0 and predicted == 1:
            FP += 1
        elif actual == 1 and predicted == 0:
            FN += 1

    return TP, TN, FP, FN


def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return (TP + TN) / len(y_true)


def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)


def specificity(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    if TN + FP == 0:
        return 0
    return TN / (TN + FP)


def error_rate(y_true, y_pred):
    return 1 - accuracy(y_true, y_pred)


def multi_confusion_matrix(y_true, y_pred):
    classes = sorted(list(set(y_true)))
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}

    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    return matrix

def multi_class_metrics(y_true, y_pred):
    classes = sorted(list(set(y_true)))
    cm = multi_confusion_matrix(y_true, y_pred)

    metrics = {}

    total_samples = len(y_true)

    macro_p = macro_r = macro_f1 = 0
    micro_TP = micro_FP = micro_FN = 0

    for c in classes:
        TP = cm[c][c]
        FP = sum(cm[other][c] for other in classes if other != c)
        FN = sum(cm[c][other] for other in classes if other != c)
        support = sum(cm[c].values())

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        micro_TP += TP
        micro_FP += FP
        micro_FN += FN

    macro_avg = {
        "precision": macro_p / len(classes),
        "recall": macro_r / len(classes),
        "f1": macro_f1 / len(classes)
    }

    micro_precision = micro_TP / (micro_TP + micro_FP)
    micro_recall = micro_TP / (micro_TP + micro_FN)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    micro_avg = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1
    }

    return metrics, macro_avg, micro_avg

def mse(y_true, y_pred):
    return sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / len(y_true)

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return sum(abs(a - p) for a, p in zip(y_true, y_pred)) / len(y_true)

def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((y - mean_y) ** 2 for y in y_true)
    ss_residual = sum((a - p) ** 2 for a, p in zip(y_true, y_pred))
    return 1 - (ss_residual / ss_total)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(vector):
    exp_vals = [math.exp(x) for x in vector]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]

def binary_log_loss(y_true, y_prob):
    epsilon = 1e-15
    loss = 0

    for y, p in zip(y_true, y_prob):
        p = max(min(p, 1 - epsilon), epsilon)
        loss += y * math.log(p) + (1 - y) * math.log(1 - p)

    return -loss / len(y_true)

def cross_entropy(y_true, y_prob):
    epsilon = 1e-15
    loss = 0

    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            p = max(min(y_prob[i][j], 1 - epsilon), epsilon)
            loss += y_true[i][j] * math.log(p)

    return -loss / len(y_true)

def roc_curve_points(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    points = []

    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in y_scores]
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)

        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0

        points.append((FPR, TPR))

    return points

def auc_score(roc_points):
    roc_points = sorted(roc_points)
    area = 0

    for i in range(1, len(roc_points)):
        x1, y1 = roc_points[i-1]
        x2, y2 = roc_points[i]
        area += (x2 - x1) * (y1 + y2) / 2

    return area




data = [10, 12, 14, 16, 18]
v1 = [1, 2, 3]
v2 = [4, 5, 6]

print("Mean:", mean(data))
print("Variance:", Variance(data))
print("Standard Deviation:", std_dev(data))
print("Maximum:", maximum(data))
print("Minimum:", minimum(data))
print("Custom Sum:", custom_sum(data))
print("Vector Add:", vec_add(v1, v2))
print("Vector Subtract:", vec_sub(v1, v2))
print("Scalar Multiply:", Scalar_multiplication(3, v1))
print("Z-Scores:", z_scores(data))
print("25th Percentile:", percentile(data, 25))
print("50th Percentile:", percentile(data, 50))
print("75th Percentile:", percentile(data, 75))

Q1, Q2, Q3 = quartiles(data)
print("Quartiles:", Q1, Q2, Q3)

# ------------------------------
# Testing Classification Metrics
# ------------------------------

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

TP, TN, FP, FN = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("Accuracy:", accuracy(y_true, y_pred))
print("Precision:", precision(y_true, y_pred))
print("Recall:", recall(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Specificity:", specificity(y_true, y_pred))
print("Error Rate:", error_rate(y_true, y_pred))

# ------------------------------
# Testing Regression Metrics
# ------------------------------

y_true_reg = [3, -0.5, 2, 7]
y_pred_reg = [2.5, 0.0, 2, 8]

print("\nRegression Metrics:")
print("MSE:", mse(y_true_reg, y_pred_reg))
print("RMSE:", rmse(y_true_reg, y_pred_reg))
print("MAE:", mae(y_true_reg, y_pred_reg))
print("R2 Score:", r2_score(y_true_reg, y_pred_reg))