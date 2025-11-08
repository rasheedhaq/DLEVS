"""
train.py
Training loop for DLEVS models.
"""
import logging
import tensorflow as tf

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, optimizer):
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    logging.info(f"Training complete. Final val_loss: {history.history['val_loss'][-1]}")
    return history
