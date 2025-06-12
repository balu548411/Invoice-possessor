import tensorflow as tf
from tensorflow.keras import layers, Model, applications
from transformers import TFBertModel, BertTokenizer
import numpy as np

class InvoiceProcessor:
    def __init__(self, img_size=(800, 800), max_text_length=512, num_fields=10):
        self.img_size = img_size
        self.max_text_length = max_text_length
        self.num_fields = num_fields  # Number of fields to extract
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Build the model
        self.model = self.build_model()
        
    def build_model(self):
        """Build a model that combines image and text features"""
        # 1. Image input branch using EfficientNet for feature extraction
        image_input = layers.Input(shape=(self.img_size[0], self.img_size[1], 3), name='image_input')
        
        # Use a pre-trained CNN as feature extractor
        base_model = applications.EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_tensor=image_input
        )
        
        # Freeze early layers to prevent overfitting
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        # Add global pooling to get a fixed-size feature vector
        image_features = layers.GlobalAveragePooling2D()(base_model.output)
        image_features = layers.Dropout(0.3)(image_features)
        image_features = layers.Dense(512, activation='relu')(image_features)
        
        # 2. Text input branch using BERT for text understanding
        # Input for text tokens, attention mask, and token type IDs
        input_ids = layers.Input(shape=(self.max_text_length,), dtype=tf.int32, name='input_ids')
        attention_mask = layers.Input(shape=(self.max_text_length,), dtype=tf.int32, name='attention_mask')
        token_type_ids = layers.Input(shape=(self.max_text_length,), dtype=tf.int32, name='token_type_ids')
        
        # BERT layer
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers to prevent overfitting (optional)
        bert_model.trainable = False
        
        bert_output = bert_model([input_ids, attention_mask, token_type_ids])[0]
        
        # Use CLS token output as document representation
        text_features = bert_output[:, 0, :]
        text_features = layers.Dropout(0.3)(text_features)
        text_features = layers.Dense(512, activation='relu')(text_features)
        
        # 3. Combine features from both branches
        combined_features = layers.concatenate([image_features, text_features])
        combined_features = layers.Dense(1024, activation='relu')(combined_features)
        combined_features = layers.Dropout(0.5)(combined_features)
        combined_features = layers.Dense(512, activation='relu')(combined_features)
        
        # 4. Outputs for different fields
        # We'll create multiple outputs for different invoice fields
        # Each output will be a head with character-level prediction
        outputs = {}
        
        # Common fields in invoices
        field_names = [
            'invoice_number', 'date', 'due_date', 'total_amount', 
            'vendor_name', 'customer_name', 'tax_amount', 
            'subtotal', 'payment_terms', 'description'
        ]
        
        for field in field_names[:self.num_fields]:
            # For each field, we predict a probability of each character class
            field_out = layers.Dense(256, activation='relu')(combined_features)
            field_out = layers.Dense(128, activation='relu')(field_out)
            
            # For simplicity, we'll predict each field as a 50-character sequence with vocab size 100
            # In a real system, you would use a more sophisticated approach
            field_out = layers.Dense(50 * 100, activation='linear')(field_out)
            field_out = layers.Reshape((50, 100))(field_out)
            field_out = layers.Softmax(axis=-1)(field_out)
            
            outputs[field] = field_out
        
        # Create the model with multiple inputs and outputs
        model = Model(
            inputs=[image_input, input_ids, attention_mask, token_type_ids],
            outputs=outputs
        )
        
        # Compile with loss and metrics
        losses = {field: 'categorical_crossentropy' for field in field_names[:self.num_fields]}
        loss_weights = {field: 1.0 for field in field_names[:self.num_fields]}
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=losses,
            loss_weights=loss_weights,
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_batch(self, images, text_data):
        """Prepare a batch of data for training"""
        # Tokenize text data
        encodings = self.tokenizer(
            text_data,
            truncation=True,
            padding='max_length',
            max_length=self.max_text_length,
            return_tensors='tf'
        )
        
        # Create BERT inputs
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        token_type_ids = encodings['token_type_ids']
        
        return {
            'image_input': np.array(images),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        
    def train(self, train_images, train_text, val_images, val_text, train_labels, val_labels, epochs=10, batch_size=8):
        """Train the model"""
        # Prepare data
        train_inputs = self.prepare_batch(train_images, train_text)
        val_inputs = self.prepare_batch(val_images, val_text)
        
        # Convert labels to one-hot encoding (simplified)
        # In a real system, this would be more complex based on your label format
        train_outputs = {}
        val_outputs = {}
        
        field_names = [
            'invoice_number', 'date', 'due_date', 'total_amount', 
            'vendor_name', 'customer_name', 'tax_amount', 
            'subtotal', 'payment_terms', 'description'
        ]
        
        for idx, field in enumerate(field_names[:self.num_fields]):
            # This is a simplified placeholder - you would need to convert your actual labels
            # to the appropriate format for each field
            zeros = np.zeros((len(train_labels), 50, 100))
            train_outputs[field] = zeros
            
            zeros_val = np.zeros((len(val_labels), 50, 100))
            val_outputs[field] = zeros_val
        
        # Create callbacks for model checkpoints and early stopping
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='model_checkpoints/model_{epoch:02d}_{val_loss:.2f}.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            train_inputs,
            train_outputs,
            validation_data=(val_inputs, val_outputs),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_callback, early_stopping]
        )
        
        return history
    
    def predict(self, image, text):
        """Make predictions on a single invoice"""
        # Prepare input
        inputs = self.prepare_batch([image], [text])
        
        # Get model predictions
        predictions = self.model.predict(inputs)
        
        # Process predictions
        results = {}
        field_names = [
            'invoice_number', 'date', 'due_date', 'total_amount', 
            'vendor_name', 'customer_name', 'tax_amount', 
            'subtotal', 'payment_terms', 'description'
        ]
        
        for idx, field in enumerate(field_names[:self.num_fields]):
            # Convert from one-hot predictions to text (simplified)
            # In a real system, this would be more complex
            char_indices = np.argmax(predictions[field][0], axis=1)
            
            # Map indices to characters (simplified placeholder)
            # This would be replaced with your actual character mapping
            char_mapping = {i: chr(i + 32) for i in range(100)}
            field_text = ''.join([char_mapping.get(idx, '') for idx in char_indices])
            
            # Clean up the output (remove padding, etc.)
            field_text = field_text.strip()
            
            results[field] = field_text
            
        return results
    
    def save_model(self, filepath):
        """Save model weights"""
        self.model.save_weights(filepath)
        
    def load_model(self, filepath):
        """Load model weights"""
        self.model.load_weights(filepath) 