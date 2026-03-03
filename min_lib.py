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

