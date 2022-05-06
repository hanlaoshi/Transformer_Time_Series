def prediction(model, dl, t0, future):
    # 预测前先load model， dl就是待预测数据，t0就是前n和时间点，future就是要预测的n个时间点
    # 比如你要用一周内前五天的数据训练模型，来预测后两天的值 t0 = 5 * 24 = 120， future = 48
    with torch.no_grad():
        predictions = []
        observations = []
        for step, (x, y, attention_masks) in enumerate(dl):
            # x: (batch_size， total_ts_length)
            # y: (batch_size, total_ts_length)
            # ouput:(batch_size, total_ts_length, 1)
            output = model(x, y, attention_masks[0])
            history = y[:, :t0].cpu().numpy().tolist()
            for p, o in zip(output.squeeze()[:, (t0 - 1):(t0 + future - 1)].cpu().numpy().tolist(),
                            y[:, t0:].cpu().numpy().tolist()):  # not missing data

                predictions.append(p) # (batch_size, future)
                observations.append(o) # (batch_size, future)
        num = 0
        den = 0
        for hist, y_preds, y_trues in zip(history, predictions, observations):
            plot_result(hist, y_preds, y_trues, t0)
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den
    return Rp
