import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import pandas as pd


def run_experiments(neurons_list, x_scaled, y, x_scaled_val, y_val, epochs_list, batch_size_list, validation_split,
                    optimizer_list=['adam'],
                    activation_list=['relu'], l2=0.1, patience=10, learning_rate=0.01, dropout=0,
                    output_activation='linear', loss_metric='mean_squared_error', monitor_metric='val_loss'):
    results = []

    for neurons in neurons_list:
        layers = len(neurons)

        for epochs in epochs_list:
            for batch_size in batch_size_list:
                for optimizer in optimizer_list:
                    for activation in activation_list:
                        model = Sequential()
                        model.add(Dense(neurons[0], activation=activation, input_shape=(x_scaled.shape[1],),
                                        kernel_regularizer=regularizers.l2(l2)))
                        if dropout != 0:
                            model.add(Dropout(dropout))
                        for i in range(1, layers):
                            model.add(Dense(neurons[i], activation=activation))
                            if dropout != 0:
                                model.add(Dropout(dropout))
                        model.add(Dense(1, activation=output_activation))

                        optimizer_model = optimizers.get(optimizer)
                        if learning_rate != 0:
                            optimizer_model.learning_rate = learning_rate
                        model.compile(optimizer=optimizer_model, loss=loss_metric, metrics=['accuracy'])

                        early_stopping = EarlyStopping(patience=patience, monitor=monitor_metric,
                                                       restore_best_weights=True)
                        model_checkpoint = ModelCheckpoint(
                            f'../data/models/best_model_{neurons}_{epochs}_{batch_size}_{optimizer}_{activation}_{learning_rate}_{l2}',
                            save_best_only=True)
                        tensorboard = TensorBoard(
                            log_dir=f"logs/{neurons}_{epochs}_{batch_size}_{optimizer}_{activation}_{learning_rate}_{l2}")

                        class CustomDataGenerator(Sequence):
                            def __init__(self, x, y, batch_size):
                                self.x = x
                                self.y = y
                                self.batch_size = batch_size
                                self.num_samples = len(x)

                            def __len__(self):
                                return int(np.ceil(self.num_samples / self.batch_size))

                            def __getitem__(self, index):
                                start_idx = index * self.batch_size
                                end_idx = (index + 1) * self.batch_size
                                batch_x = self.x[start_idx:end_idx]
                                batch_y = self.y[start_idx:end_idx]
                                return batch_x, batch_y

                        x_train = x_scaled
                        x_val = x_scaled_val
                        y_train = y
                        y_val = y_val

                        train_data_generator = CustomDataGenerator(x_train, y_train, batch_size)
                        val_data_generator = CustomDataGenerator(x_val, y_val, batch_size)

                        history = model.fit(train_data_generator, epochs=epochs,
                                            validation_data=val_data_generator,
                                            callbacks=[early_stopping, model_checkpoint, tensorboard])

                        def visualize_loss(history, title):
                            loss = history.history["loss"]
                            val_loss = history.history["val_loss"]
                            epochs = range(len(loss))
                            plt.figure()
                            plt.plot(epochs, loss, "b", label="Training loss")
                            plt.plot(epochs, val_loss, "r", label="Validation loss")
                            plt.title(title)
                            plt.xlabel("Epochs")
                            plt.ylabel("Loss")
                            plt.legend()
                            plt.show()

                        visualize_loss(history, "Training and Validation Loss")

                        # Predictions on validation data
                        y_val_pred = model.predict(val_data_generator)
                        y_val_pred_binary = np.round(y_val_pred)

                        # Binary classification metrics
                        accuracy = accuracy_score(y_val, y_val_pred_binary)
                        precision = precision_score(y_val, y_val_pred_binary)
                        recall = recall_score(y_val, y_val_pred_binary)
                        f1 = f1_score(y_val, y_val_pred_binary)

                        print(f"Accuracy on validation data: {accuracy}")
                        print(f"Precision on validation data: {precision}")
                        print(f"Recall on validation data: {recall}")
                        print(f"F1 Score on validation data: {f1}")

                        results.append({'Neurons': neurons, 'Layers': layers, 'Train Loss': history.history['loss'][-2],
                                        'Validation Loss': history.history['val_loss'][-2],
                                        'Optimizer': optimizer, 'Activation': activation, 'Epochs': epochs,
                                        'Batch Size': batch_size,
                                        'Validation Split': validation_split,
                                        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1,
                                        'Regularizer': l2})

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/models/results.csv', index=False)
    results_df.to_csv('data/models/results_all.csv', mode='a', index=False, header=False)

    return results_df,model
