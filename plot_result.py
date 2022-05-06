def plot_result(history, yhat, ytruth, t0):
    # 带上历史值
    yhat = history + yhat
    ytruth = history + ytruth
    # 画图
    x = range(len(ytruth))
    yhat = np.round(yhat, 2)
    ytruth = np.round(ytruth, 2)
    plt.figure(facecolor='w')  
    plt.plot(range(len(x)), ytruth, 'green', linewidth=1.5, label='ground truth')
    plt.plot(range(len(x)), yhat, 'blue', alpha=0.8, linewidth=1.2, label='predict value')
    # 画条预测起始线
    plt.vlines(t0, yhat.min() * 0.99, yhat.max() * 1.01,
               alpha=0.7, colors="r", linestyles="dashed")
    # plt.text(0.15, 0.01, error_message, size=10, alpha=0.9, transform=plt.gca().transAxes)  # 相对位置，经验设置值
    plt.legend(loc='best')  # 设置标签的位置
    plt.grid(True)
    plt.show()
