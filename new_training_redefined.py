import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import glob
import kerastuner as kt
from tqdm import tqdm
import librosa
import matplotlib.pyplot as plt

AUDIO_PATH = "audio_files1"
TUNING_EPOCHS = 15
SAMPLE_RATE = 44100
DURATION = 2
BATCH_SIZE = 16  
EPOCHS = 50  
PATIENCE = 8  
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4


TRAINING_AUGMENT_PROB = 0.3




NOTES = ['A0', 'A#0', 'B0']
for i in range(1, 8):
    NOTES.extend([f'C{i}', f'C#{i}', f'D{i}', f'D#{i}', f'E{i}', f'F{i}', f'F#{i}', f'G{i}', f'G#{i}', f'A{i}', f'A#{i}', f'B{i}'])
NOTES.append('C8')

NOTES_Dictionary = { key:value for value,key in enumerate(NOTES) }

audio_paths = glob.glob(f"{AUDIO_PATH}/*/*.wav")
print(f"Total Audios Founds {len(audio_paths)}")
note_name = (os.path.dirname(audio_paths[10000]).split(os.sep)[-1]).replace("s","#")

name_array = note_name.split("_")
print([NOTES_Dictionary[note_name] for note_name in name_array[:-1 if len(name_array) > 1 else 1]])
# [NOTES_Dictionary[note_name] for note_name in name_array[:-1 if len(name_array) > 1 else 1]]




# class DataGenerator(Sequence):
#     def __init__(self, data_paths,batch_size=BATCH_SIZE, audio_length=44100*2, sr=44100,shuffle=True):
#         self.data_paths = data_paths
#         self.batch_size = batch_size
#         self.audio_length = audio_length
#         self.shuffle = shuffle
#         self.sr = sr
#         self.on_epoch_end()
    
#     def __len__(self):
#         return int(np.ceil(len(self.data_paths) / self.batch_size))
    
#     def __getitem__(self, idx):
#         batch_paths = self.data_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        
#         batch_x = []
#         batch_y = []
#         for fp in batch_paths:
#             audio, _ = librosa.load(fp, sr=self.sr, mono=True)

            
#             if len(audio) < self.audio_length:
#                 pad_width = self.audio_length - len(audio)
#                 audio = np.pad(audio, (0, pad_width))
#             else:
#                 audio = audio[:self.audio_length]

#             # Normalize audio for MelSpectrogram input
#             audio = audio.astype(np.float32) / np.max(np.abs(audio))  
            
#             batch_x.append(audio)
#             batch_y.append(self.generate_label(fp))
#         return np.array(batch_x), np.array(batch_y)
    
#     def on_epoch_end(self):
#         if self.shuffle:
#             # Shuffle data only for the training generator
#             indices = np.arange(len(self.data_paths))
#             np.random.shuffle(indices)
#             self.data_paths = [self.data_paths[i] for i in indices]

#     def generate_label(self,audio_path):
        
#         note_name = (os.path.dirname(audio_path).split(os.sep)[-1])
#         if 'silence'  in note_name:
#             return np.int16(np.zeros(88))
#         note_name = note_name.replace("s","#")

#         name_array = note_name.split("_")
#         return [1 if i in [NOTES_Dictionary[note_name] for note_name in name_array[:-1 if len(name_array) > 1 else 1]] else 0 for i in range(88)]

    
class DataGenerator(Sequence):
    def __init__(self, data_paths, batch_size=BATCH_SIZE, audio_length=44100*2, 
                 sr=44100, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.audio_length = audio_length
        self.shuffle = shuffle
        self.sr = sr
        self.augment = augment
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.data_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_paths = self.data_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        for fp in batch_paths:
            try:
                audio, _ = librosa.load(fp, sr=self.sr, mono=True)
                
                # Length adjustment
                if len(audio) < self.audio_length:
                    pad_width = self.audio_length - len(audio)
                    audio = np.pad(audio, (0, pad_width))
                else:
                    audio = audio[:self.audio_length]

               
                if self.augment and np.random.random() < TRAINING_AUGMENT_PROB:
                    audio = self.apply_light_augmentation(audio)
                

                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio.astype(np.float32) / max_val
                else:
                    audio = audio.astype(np.float32)
                
                batch_x.append(audio)
                batch_y.append(self.generate_label(fp))
                
            except Exception as e:
                print(f"Error loading {fp}: {e}")
                # Skip this sample or use a silent sample
                audio = np.zeros(self.audio_length, dtype=np.float32)
                batch_x.append(audio)
                batch_y.append(np.zeros(88, dtype=np.float32))
        
        return np.array(batch_x), np.array(batch_y)
    
    def apply_light_augmentation(self, audio):
        """Light augmentation during training"""

        if np.random.random() < 0.5:
            shift = np.random.randint(-1000, 1000)
            audio = np.roll(audio, shift)
        

        if np.random.random() < 0.5:
            volume_factor = np.random.uniform(0.8, 1.2)
            audio = audio * volume_factor
        

        if np.random.random() < 0.3:
            noise_factor = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_factor, audio.shape)
            audio = audio + noise
        
        return audio
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.data_paths))
            np.random.shuffle(indices)
            self.data_paths = [self.data_paths[i] for i in indices]

    def generate_label(self, audio_path):
        note_name = os.path.dirname(audio_path).split(os.sep)[-1]
        
        if 'silence' in note_name:
            return np.zeros(88, dtype=np.float32)
        
        note_name = note_name.replace("s", "#")
        name_array = note_name.split("_")
        

        chord_indicators = ['Major', 'Minor', 'Diminished', 'Augmented', 'Suspended2', 
                           'Suspended4', 'Major7th', 'Minor7th', 'Dominant7th',
                           'MinorMajor7th', 'Diminished7th', 'Major6th', 'Minor6th',
                           'Add9', 'MinorAdd9']
        

        chord_start_idx = len(name_array)
        for i, part in enumerate(name_array):
            if any(chord in part for chord in chord_indicators):
                chord_start_idx = i
                break
        

        note_names = name_array[:chord_start_idx]
        
        label = np.zeros(88, dtype=np.float32)
        for note_name in note_names:
            if note_name in NOTES_Dictionary:
                label[NOTES_Dictionary[note_name]] = 1
        
        return label

# Custom F1 Score metric for multi-label classification
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()




X_train_val, X_test,  = train_test_split(
    audio_paths, test_size=0.2, random_state=42
)
# Then, split the 80% into actual train and validation sets
X_train, X_val  = train_test_split(
    X_train_val, test_size=0.25, random_state=42 
)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# --- Create Data Generators ---
train_generator = DataGenerator(X_train, )
val_generator = DataGenerator(X_val)
test_generator = DataGenerator(X_test, shuffle=False)
audio_length = SAMPLE_RATE*DURATION

# # former model

input = Input(shape =(SAMPLE_RATE*DURATION,))
x = layers.MelSpectrogram(
        fft_length=2048,
        sequence_stride=512,
        sampling_rate=44100,
        num_mel_bins=128,
        power_to_db=True,
        
    )(input)
x = layers.Reshape((x.shape[1],x.shape[2],1))(x)
x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
# x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.5)(x)
# x = layers.Dense(1024,activation='relu')(x)
# x = layers.Dropout(0.8)(x)
x = layers.Dropout(0.3)(x)
x = layers.Flatten()(x)
output = layers.Dense(88,activation = "sigmoid")(x)
model = Model(input,output)


model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), # Lower initial learning rate
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), # This is binary accuracy in this context
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            F1Score(name='f1_score'),
        ]
    )
model.summary()


callbacks = [
EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, restore_best_weights=True),
ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
]


history = model.fit(
train_generator,
validation_data=val_generator,
epochs=EPOCHS,
callbacks=callbacks
)
    



test_results = model.evaluate(test_generator, verbose=1)
print("Test Results:", dict(zip(model.metrics_names, test_results)))


y_true_all = []
y_pred_all = []

for i in tqdm(range(len(test_generator)), desc="Collecting predictions"):
    batch_x, batch_y = test_generator[i]
    batch_pred = model.predict(batch_x, verbose=0)
    y_true_all.extend(batch_y)
    y_pred_all.extend(batch_pred)

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# Convert predictions to binary
y_pred_binary = (y_pred_all > 0.5).astype(int)

problematic_notes = []
good_notes = []

for i, note in enumerate(NOTES):
    note_precision = tf.keras.metrics.Precision()
    note_recall = tf.keras.metrics.Recall()
    
    note_precision.update_state(y_true_all[:, i], y_pred_binary[:, i])
    note_recall.update_state(y_true_all[:, i], y_pred_binary[:, i])
    
    p = note_precision.result().numpy()
    r = note_recall.result().numpy()
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    
    if f1 < 0.5:  # Problematic notes
        problematic_notes.append(f"{note}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    else:
        good_notes.append(f"{note}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

if problematic_notes:
    print("Notes with F1 < 0.5:")
    for note_info in problematic_notes[:10]:  # Show first 10
        print(note_info)
else:
    print("All notes have F1 >= 0.5!")

print(f"\nGood performing notes: {len(good_notes)}")
print(f"Problematic notes: {len(problematic_notes)}")

# Plot training history
try:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['f1_score'], label='Training F1')
    plt.plot(history.history['val_f1_score'], label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Could not create plots: {e}")


# #hyper parameter tunning

# def build_model(hp):
#     audio_length = SAMPLE_RATE * DURATION
    
#     inputs = Input(shape=(audio_length,))
#     x = layers.MelSpectrogram(
#         fft_length=2048,
#         sequence_stride=512,
#         sampling_rate=44100,
#         num_mel_bins=128,
#         power_to_db=True,
#     )(inputs)
#     x = layers.Reshape((x.shape[1], x.shape[2], 1))(x)

#     # --- Tunable Hyperparameters ---
#     hp_filters = hp.Choice('filters', values=[32, 64, 128])
#     hp_kernel_size = hp.Choice('kernel_size', values=[3, 5])
#     hp_dense_units = hp.Int('dense_units', min_value=256, max_value=1024, step=256)
#     hp_dropout = hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1)
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#     # --------------------------------

#     x = layers.Conv2D(hp_filters, (hp_kernel_size, hp_kernel_size), activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Dropout(hp_dropout)(x)
    
#     x = layers.Flatten()(x)
#     x = layers.Dense(units=hp_dense_units, activation='relu')(x)
#     x = layers.Dropout(hp_dropout)(x)

#     outputs = layers.Dense(88, activation="sigmoid")(x)
#     model = Model(inputs, outputs)

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
#         loss='binary_crossentropy',
#         metrics=['binary_accuracy', 'precision', 'recall']
#     )
#     return model



# tuner = kt.Hyperband(
#     build_model,
#     objective='val_loss',
#     max_epochs=TUNING_EPOCHS,
#     factor=3,
#     directory='keras_tuner_dir',
#     project_name='piano_note_tuning'
# )

# # --- 3. Run the Hyperparameter Search ---
# print("\nStarting hyperparameter search...")
# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# tuner.search(train_generator, epochs=TUNING_EPOCHS, validation_data=val_generator, callbacks=[stop_early])

# # --- 4. Get the Best Model ---
# print("\nHyperparameter search complete. Retrieving the best model...")
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# print(f"""
# The optimal number of filters is {best_hps.get('filters')},
# the optimal kernel size is {best_hps.get('kernel_size')},
# the optimal number of dense units is {best_hps.get('dense_units')},
# the optimal dropout rate is {best_hps.get('dropout')},
# and the optimal learning rate is {best_hps.get('learning_rate')}.
# """)

# model_best = tuner.get_best_models(num_models=1)[0]

# # --- 5. Evaluate and Convert the Best Model ---
# print("\nEvaluating the best model on the test set...")
# model_best.evaluate(test_generator)


print("\nConverting the final, best model to TFLite format for Flutter...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS,
tf.lite.OpsSet.SELECT_TF_OPS
]
# converter.allow_custom_ops = True
tflite_model = converter.convert()

with open('piano_note_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\nSuccess! 'piano_note_model.tflite' is ready for your Flutter app.")



