import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras for CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# For image processing
from PIL import Image
import cv2

class PitchTypeClassifier:
    def __init__(self, data_folder='dataset/images', batch_size=32, img_size=(224, 224)):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        self.model = None

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

    def extract_pitch_type_from_filename(self, filename):
        """Extract pitch type from filename like '2018-04-01_529450_atbat_13_pitch_1_ST_Sweeper_none_frame_0.jpg'"""
        try:
            # Split by '_' and find pitch type (usually 6th element from start)
            parts = filename.split('_')
            if len(parts) >= 7:
                # Look for pitch type codes: FF, ST, SL, CU, FS, etc.
                for part in parts:
                    if part in ['FF', 'ST', 'SL', 'CU', 'FS', 'FC', 'SI', 'CH', 'KN', 'UNK']:
                        return part
            return 'UNK'  # Unknown pitch type
        except:
            return 'UNK'

    def create_dataset_csv(self):
        """Create a CSV file mapping images to their pitch types"""
        print("Creating dataset CSV...")

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(self.data_folder, '**', ext), recursive=True))

        print(f"Found {len(image_files)} images")

        # Create dataset list
        dataset = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            pitch_type = self.extract_pitch_type_from_filename(filename)
            dataset.append({
                'filepath': img_path,
                'filename': filename,
                'pitch_type': pitch_type
            })

        # Convert to DataFrame
        df = pd.DataFrame(dataset)

        # Filter out unknown pitch types and save
        df = df[df['pitch_type'] != 'UNK']
        df.to_csv('../data/processed/pitch_type_dataset.csv', index=False)

        print(f"Dataset created with {len(df)} samples")
        print("Pitch type distribution:")
        print(df['pitch_type'].value_counts())

        return df

    def prepare_data_generators(self, df, test_size=0.2, val_size=0.2):
        """Prepare train/validation/test data generators"""
        print("Preparing data generators...")

        # Encode labels
        df['pitch_type_encoded'] = self.label_encoder.fit_transform(df['pitch_type'])

        # Split data
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['pitch_type'], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), stratify=train_df['pitch_type'], random_state=42)

        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )

        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filepath',
            y_col='pitch_type',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_generator = val_test_datagen.flow_from_dataframe(
            val_df,
            x_col='filepath',
            y_col='pitch_type',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = val_test_datagen.flow_from_dataframe(
            test_df,
            x_col='filepath',
            y_col='pitch_type',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, val_generator, test_generator, train_df, val_df, test_df

    def build_model(self, num_classes):
        """Build CNN model for pitch type classification"""
        print(f"Building model for {num_classes} pitch types...")

        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(self.model.summary())
        return self.model

    def train_model(self, train_generator, val_generator, epochs=50):
        """Train the model"""
        print("Starting model training...")

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max'
        )

        model_checkpoint = ModelCheckpoint(
            '../models/pitch_classifier/best_pitch_classifier.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

        # Train
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        print("Training completed!")
        return history

    def evaluate_model(self, test_generator, test_df):
        """Evaluate model performance"""
        print("Evaluating model...")

        # Predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes

        # Convert back to pitch type labels
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        true_labels = self.label_encoder.inverse_transform(true_classes)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels))

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Pitch Type Classification')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('../models/pitch_classifier/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return predicted_labels, true_labels

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('../models/pitch_classifier/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_complete_pipeline(self, max_samples=None):
        """Run the complete ML pipeline"""
        print("Starting Pitch Type Classification Pipeline")
        print("=" * 60)

        if max_samples:
            print(f"WARNING: Using only {max_samples} samples for testing")

        # Step 1: Create dataset
        df = self.create_dataset_csv()

        # Apply sample limit if specified
        if max_samples and len(df) > max_samples:
            print(f"Reducing dataset from {len(df)} to {max_samples} samples...")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Check if we have enough data
        if len(df) < 10:
            print("ERROR: Not enough training data. Need at least 10 samples.")
            return

        # Step 2: Prepare data generators
        train_gen, val_gen, test_gen, train_df, val_df, test_df = self.prepare_data_generators(df)

        # Step 3: Build model
        num_classes = len(self.label_encoder.classes_)
        self.build_model(num_classes)

        # Step 4: Train model
        history = self.train_model(train_gen, val_gen)

        # Step 5: Evaluate model
        predictions, true_labels = self.evaluate_model(test_gen, test_df)

        # Step 6: Plot results
        self.plot_training_history(history)

        print("\nPipeline completed successfully!")
        print(f"Final test accuracy: {self.model.evaluate(test_gen)[1]:.4f}")
        print("Saved files: ../models/pitch_classifier/best_pitch_classifier.h5, ../models/pitch_classifier/confusion_matrix.png, ../models/pitch_classifier/training_history.png")

        return history, predictions, true_labels

if __name__ == "__main__":
    # Initialize classifier
    classifier = PitchTypeClassifier()

    # Run complete pipeline with limited samples for testing
    # Change None to a number (e.g., 500) to limit samples for faster testing
    max_samples = 500  # Set to None for full dataset

    results = classifier.run_complete_pipeline(max_samples=max_samples)
