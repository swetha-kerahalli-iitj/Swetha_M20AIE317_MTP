from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def rayleigh_qam_16(data, config, smell, out_folder=OUT_FOLDER, dim=DIM, iteration=0, is_final=False):
    tf.keras.backend.clear_session()
    max_features = int(max(np.max(data.train_data), np.max(data.eval_data)))
    print("max features: " + str(max_features))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=max_features + 1,
                                        output_dim=config.emb_output,
                                        mask_zero=True))
    for i in range(0, config.layers - 1):
        model.add(tf.keras.layers.LSTM(config.lstm_units, return_sequences=True, recurrent_dropout=0.1, dropout=0.1))
    # model.add(tf.keras.layers.Dropout(config.dropout))
    model.add(tf.keras.layers.LSTM(config.lstm_units, recurrent_dropout=0.1, dropout=0.1))
    model.add(tf.keras.layers.Dropout(config.dropout))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0.0001,
                                                 patience=2,
                                                 verbose=1,
                                                 mode='auto')
    best_model_filepath = 'weights_best.rnn.' + smell + str(iteration) + '.hdf5'
    if os.path.exists(best_model_filepath):
        print("deleting the old weights file..")
        os.remove(best_model_filepath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_filepath, monitor='val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks_list = [earlystop, checkpoint]

    batch_sizes = [32, 64, 128, 256]
    b_size = int(len(data.train_labels) / 512)
    if b_size > len(batch_sizes) - 1:
        b_size = len(batch_sizes) - 1

    if is_final:
        model.fit(data.train_data,
                  data.train_labels,
                  epochs=config.epochs,
                  batch_size=batch_sizes[b_size])
        stopped_epoch = config.epochs

    else:
        model.fit(data.train_data,
                  data.train_labels,
                  validation_split=0.2,
                  epochs=config.epochs,
                  batch_size=batch_sizes[b_size],
                  callbacks=callbacks_list)
        stopped_epoch = earlystop.stopped_epoch
        model.load_weights(best_model_filepath)

    # y_pred = model.predict(data.eval_data).ravel()
    # y_pred = model.predict_classes(data.eval_data)
    # We manually apply classification threshold
    prob = model.predict(data.eval_data)
    y_pred = inputs.get_predicted_y(prob, CLASSIFIER_THRESHOLD)

    auc, accuracy, precision, recall, f1, average_precision, fpr, tpr = \
        metrics_util.get_all_metrics(model, data.eval_data, data.eval_labels, y_pred)

    if is_final:
        plot_util.save_roc_curve(fpr, tpr, auc, smell, config, out_folder, DIM)
        plot_util.save_precision_recall_curve(data.eval_labels, y_pred, average_precision, smell, config, out_folder,
                                              dim, "rnn")
    tf.keras.backend.clear_session()
    return auc, accuracy, precision, recall, f1, average_precision, stopped_epoch


def start_training(data, config, conn, smell):
    try:
        return embedding_lstm(data, config, conn, smell)
    except Exception as ex:
        print(ex)
        return ([-1, -1, -1])


n ([-1, -1, -1])