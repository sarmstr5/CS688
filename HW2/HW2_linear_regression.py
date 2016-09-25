# Problem 1 part b
# Normal equation coefs = (Xtrans * X)inverted * Xtrans*y
from pylab import *
from datetime import datetime as dt
import pandas as pd
import numpy as np
import random

def run_descent(x, y, w, alpha, iterations):
    loss_steps = []
    current_loss = 0
    for i in range(iterations):
        loss_dict = {}
        current_loss, w = gradient_descent(x, y, w, alpha)
        # I am adding this data to a pandas df
        loss_dict[i] = [alpha, i, current_loss, w[0], w[1]]
        loss_steps.append(loss_dict[i])
    return loss_steps, w


def loss_function(x, w, t):
    N = len(x)
    total_loss = 0
    for i in range(0, N):
        total_loss += (w[0] + x[i] * w[1] - y[i]) ** 2
    return 1 / N * total_loss


def gradient_descent(x, y, w, alpha):
    N = len(x)
    w0_gradient = 0
    w1_gradient = 0
    for i in range(0, N):
        w0_gradient += (1 / N) * (w[0] + w[1] * x[i] - y[i])
        w1_gradient += (1 / N) * (w[0] + w[1] * x[i] - y[i]) * x[i]
    w[0] += -alpha * w0_gradient
    w[1] += -alpha * w1_gradient
    error = loss_function(x, w, y)
    return error, w


def find_alpha(alpha, max_alpha, step):
    loss_list = []
    while (alpha < max_alpha):
        alpha_loss_list, w = run_descent(x_pop, y, w, alpha, iterations)
        loss_list += alpha_loss_list
        alpha += step
    return loss_list


def graph_GD(loss_df):
    grouped_loss_df = loss_df.groupby(by=['alpha'])
    print(grouped_loss_df.head())
    fig, ax = plt.subplots(1, 1)
    grouped_loss_df.plot(x='iteration', y='loss', ax=ax)
    plt.legend([v[0] for v in loss_df.groupby('alpha')])


def problem_one_a(x,y):
    figure()
    plt.scatter(x=x['city_population'], y=y)
    
def problem_one_b(x, y, alpha, run_alpha, iterations):
    w = np.ones(2)
    w[0] = random.randint(-10000, 10000)
    w[1] = random.randint(-10000, 10000)
    random.rand
    max_alpha = .03
    loss_list = []
    if(run_alpha):
        loss_list = find_alpha(alpha, max_alpha, iterations)
    else:
        loss_list = run_descent(x, y, w, alpha, i, iterations)

    loss_df = pd.DataFrame(loss_list, columns=['alpha', 'iteration', 'loss', 'w1', 'w0'])
    graph_GD(loss_df)

#graph Loss vs the parameters
def problem_one_c(x,y,alpha,iterations):
    w = np.ones(2)
    # w[0] = random.randint(-10000, 10000)
    # w[1] = random.randint(-10000, 10000)
    loss_gradient, w = run_descent(w,y,w,alpha,iterations)
    loss_df = pd.DataFrame(loss_gradient, columns=['alpha', 'iteration', 'loss', 'w1', 'w0'])
    graph_df = loss_df[['iteration', 'loss', 'w1', 'w0']]
    # graph_df.plot(x='iteration', y='loss')
    figure(1)
    plt.plot(y=loss_df['w0'], x=loss_df['iteration'])
    plt.plot(y=loss_df['w1'], x=loss_df['iteration'])
    
    # plt.plot(x=graph_df['iteration'], y=graph_df['w0'])
def problem_one_d():
    pass
    validation_file = "cross_validation_results" + hour + minute + '.csv'
    error_rate = {}
    test_predictions = []
    while (attribute < stopping_value):
        cross_validated_test_set = []
        print("on {0} and the time is: {1}".format(attribute, time))
        for i in range(partitions):
            # splitting into test/train partitions
            print('this is attribute: {0} \t this is partition: {1}'.format(attribute, i))
            left_i, right_i = get_partition_indices(i, partition_size, partitions)
            train_partition, test_partition = get_cross_validation_lists(training_set, left_i, right_i)

            # predicting their sentiment
            if (multi_threading):
                pool = ThreadPool(processes=1)
                async_result = pool.apply_async(k_NN, (attribute, train_partition, test_partition))
                test_predictions = async_result.get()
            else:
                test_predictions = k_NN(attribute, train_partition, test_partition)

            print(test_predictions[0])
            cross_validated_test_set = cross_validated_test_set + test_predictions

        error_rate = calculate_prediction_error(cross_validated_test_set)
        time = dt.now()
        k += 10

        with open(validation_file, 'a') as csv:
            csv.write('{0}\t{1}\n'.format(attribute, error_rate))

if __name__ == '__main__':
    df = pd.read_csv('ex1data1.txt', names=['city_population', 'profit'])
    df.insert(1, 'ones', 1)
    x = df[['ones', 'city_population']]
    y = df['profit']
    alpha = 0.0015
    iterations = 500
    #problem_one_a(x,y)
    #problem_one_b(x,y,alpha,iterations)
    # problem_one_c(x,y,alpha,iterations)
    
    w = np.ones(2)
    w[0] = random.randint(-10, 100)
    w[1] = random.randint(-10, 100)
    loss_gradient, w = run_descent(w,y,w,alpha,iterations)
    loss_df = pd.DataFrame(loss_gradient, columns=['alpha', 'iteration', 'loss', 'w1', 'w0'])
    graph_df = loss_df[['iteration', 'loss', 'w1', 'w0']]
    # graph_df.plot(x='iteration', y='loss')
    # graph_df = graph_df.cumsum()
    plt.figure(); graph_df.plot(x='iteration', y=['w1','w0']);
    
    # print(graph_df.head())
    # grouped_loss_df = loss_df.groupby(by=['alpha'])
    # print(grouped_loss_df.head())
    # fig, ax = plt.subplots(1, 1)
    # grouped_loss_df.plot(x='iteration', y='loss', ax=ax)
    # plt.legend([v[0] for v in loss_df.groupby('alpha')])
    
    # xTx = (x.T.dot(x))
    # xTx_inv = np.linalg.inv(xTx)
    # xTy = x.T.dot(y)
    # reg_coefs = xTx_inv.dot(xTy)
    # y_hat = x.dot(reg_coefs)
    # y_hat.head()
