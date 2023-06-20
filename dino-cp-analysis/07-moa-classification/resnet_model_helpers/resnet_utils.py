import numpy as np
import pandas as pd
from tensorflow.keras import layers, regularizers, Sequential, Model, backend, optimizers, metrics, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import sys
import os
import random

def resnet_model(df_train_y, no_of_feats):
    
    """
    ResNet Model Definition
    Args:
            df_train_y: train data with only the Mechanism of actions (MOAs) target labels - pandas dataframe.
            no_of_feats: Total number of features in train data EXCLUDING the target labels
    
    Returns:
            input_: Input model layers 
            answer5: Last hidden layer in the resnet neural network
            
    For more info:https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold/blob/main/final\
    /Best%20LB/Training/2heads-ResNest-train.ipynb
    """
    input_ = layers.Input(shape=(no_of_feats, ))
    output = Sequential([layers.BatchNormalization(),layers.Dropout(0.2),
                         layers.Dense(512, activation="elu"),
                         layers.BatchNormalization(),
                         layers.Dense(256, activation="elu")])(input_)
    
    answer1 = Sequential([layers.BatchNormalization(),
                          layers.Dropout(0.3),
                          layers.Dense(512, "relu")])(output)
    
    answer2 = Sequential([layers.BatchNormalization(), layers.Dense(512, "elu"),
                          layers.BatchNormalization(), 
                          layers.Dense(256, "relu")])(layers.Concatenate()([output, answer1]))
    
    answer3 = Sequential([layers.BatchNormalization(),
                          layers.Dense(256,"elu")])(layers.Concatenate()([answer1, answer2]))
    
    answer3_ = Sequential([layers.BatchNormalization(), layers.Dense(256, "relu")
                          ])(layers.Concatenate()([answer1, answer2, answer3]))
    
    answer4 = Sequential([layers.BatchNormalization(),
                          layers.Dense(256, kernel_initializer=tf.keras.initializers.lecun_normal(seed=42),
                                       activation='selu', name='last_frozen'),layers.BatchNormalization(),
                          layers.Dense(206, kernel_initializer=tf.keras.initializers.lecun_normal(seed=42),
                                       activation='selu')])(layers.Concatenate()([output, answer2, answer3, answer3_]))
    
    answer5 = Sequential([layers.BatchNormalization(),
                          layers.Dense(df_train_y.shape[1], "sigmoid")])(answer4)
    return input_, answer5
    

def freeze_unfreeze_model_weights(model_nn, x_fold_trn, y_fold_trn, x_fold_val, y_fold_val, val_metric_old, 
                                  batch_size, model_path):
    """
    This function freezes some of the model layers that were not frozen in the original model architecture, 
    train the resulting model on train data and save the model (update the existing model) if the validation 
    loss reduces, thereafter, it unfreezes all the model layers and retrains the model on the train data, and 
    also saves the new model (update the existing model) if the validation loss reduces.
    
    The model train time depends on the 'patience' threshold.
    
    Args:
            model_nn: ResNet Model.
            x_fold_trn: K-fold train data with only phenotypic/morphological features and PCs - numpy array.
            y_fold_trn: K-fold train data with only the Mechanism of actions (MOAs) target labels - numpy array.
            x_fold_val: K-fold validation data with only phenotypic/morphological features and PCs - numpy array.
            y_fold_val: K-fold validation data with only the Mechanism of actions (MOAs) target labels - numpy array.
            val_metric_old: validation logloss prior to unfreezing/freezing of some the model layers
            Batch_size: A number that defines number of samples to work through before updating the 
            internal model parameters. The number of training examples in one forward & backward pass.
            model_path: Directory where the model will be stored
    
    Returns:
            model_nn: updated returned model after freezing/unfreezing of some of the model layers
    """
    
    label_smoothing_alpha = 0.0005
    P_MIN = label_smoothing_alpha
    P_MAX = 1 - P_MIN
    
    def logloss(y_true, y_preds):
        y_preds = tf.clip_by_value(y_preds, P_MIN, P_MAX)
        return -backend.mean(y_true * backend.log(y_preds) + (1 - y_true) * backend.log(1 - y_preds))
    
    def mean_logloss(y_preds, y_true):
        logloss = (1 - y_true) * np.log(1 - y_preds + 1e-15) + y_true * np.log(y_preds + 1e-15)
        return np.mean(-logloss)
    
    # big loop
    loop = 1
    while True:
        # Freeze_weights(model_nn, to = 'last_frozen')
        for i, layer in enumerate(model_nn.layers):
            if layer.name == "last_frozen":
                layer.trainable = True
                break
            else:
                layer.trainable = False
        
        model_nn.compile(optimizer=tf.keras.optimizers.Adadelta(lr=0.001 / 3),
                         loss=tf.losses.BinaryCrossentropy(label_smoothing=label_smoothing_alpha),
                         metrics=logloss)
        #----- Frozen Mode ------#
        reps = 0
        improved = 0
        patience = 15
        while True:
            history = model_nn.fit([x_fold_trn], y_fold_trn,epochs=5, batch_size=128,
                                   verbose=0)
            val_preds = model_nn.predict([x_fold_val])
            val_metric = mean_logloss(val_preds, y_fold_val)
            if val_metric_old - val_metric >= 1e-6:
                print('Improved:', val_metric, 'from', val_metric_old)
                reps += 5
                improved += 1
                val_metric_old = val_metric
                model_nn.save(model_path)
            elif reps < patience:
                reps += 5
                pass
            else:
                print('No Improvement, stopped')
                model_nn = tf.keras.models.load_model(model_path,custom_objects={'logloss': logloss})
                print(loop, 'loop ---> After Frozen-step best validation loss =',
                      val_metric_old, 'after', reps, 'epochs \n')
                break
        print("Total Frozen-steps improvenment:", improved)
        #----- Unfreeze all layers -----
        for i, layer in enumerate(model_nn.layers):
            layer.trainable = True
        model_nn.compile(optimizer=tf.keras.optimizers.Adadelta(lr=0.001 / 5),
                         loss=tf.losses.BinaryCrossentropy(label_smoothing=label_smoothing_alpha),
                         metrics=logloss)
        reps = 0
        improved = 0
        patience = 15
        while True:
            history = model_nn.fit([x_fold_trn], y_fold_trn,epochs=5, batch_size=batch_size,verbose=0)
            val_preds = model_nn.predict([x_fold_val])
            val_metric = mean_logloss(val_preds, y_fold_val)
            if val_metric_old - val_metric >= 1e-6:
                print('Improved:', val_metric, 'from', val_metric_old)
                reps += 5
                improved += 1
                val_metric_old = val_metric
                model_nn.save(model_path)
            elif reps < patience:
                reps += 5
                pass
            else:
                print('No Improvement, stopped')
                model_nn = tf.keras.models.load_model(model_path,custom_objects={'logloss': logloss})
                print(loop, 'loop ---> After Non-Frozen-step best validation loss =',
                      val_metric_old, 'after', reps, 'epochs \n')
                break
        print("Total Non-Frozen-steps improvenment:", improved)
        if (improved == 0):
            break
        loop += 1
            
    return model_nn
