from tensorflow import keras
#import keras
import logging
import numpy as np
import pdb

class ModelCheckpoint2(keras.callbacks.Callback):
    """Save the model after every batch
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        inputs: training inputs to evaluate loss
        outputs: training outputs to evaluate loss
        verbose: verbosity mode, 0 or 1.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, inputs, outputs, nepoch,
                 monitor='loss', verbose=1,
                 mode='min', period=1):
        super(ModelCheckpoint2, self).__init__()
        self.filepath = filepath
        self.inputs = inputs
        self.outputs = outputs
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.batches_since_last_save = 0
        self.nepoch = nepoch
        self.epochloggingth = int(nepoch/10)
        
        self.monitor_op = np.less
        self.best = np.Inf
        
        self.history = []
    
    
    def on_batch_end(self, batch, logs=None):
        self.batches_since_last_save += 1
        if self.batches_since_last_save >= self.period:
            self.batches_since_last_save = 0
            current = self.model.evaluate(self.inputs, self.outputs, verbose=0)
            self.history.append(current)
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('\nBatch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (batch + 1, self.monitor, self.best,
                                     current, self.filepath))
                self.best = current
                self.model.save_weights(self.filepath, overwrite=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%(self.epochloggingth)==0:
            logging.info('Epoch %d/%d has been reached'%(epoch+1,self.nepoch))