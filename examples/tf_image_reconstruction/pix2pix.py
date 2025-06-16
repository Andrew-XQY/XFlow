import tensorflow as tf
import os
from typing import Any, Tuple, Dict
from xflow.models.base import BaseModel

class Pix2Pix(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pix2Pix model with config.
        
        Args:
            config: Dict containing model configuration
                - channels: Number of image channels (1 for grayscale, 3 for RGB)
                - lambda_l1: Weight for L1 loss (default: 100)
                - learning_rate: Learning rate (default: 2e-4)
                - beta_1: Adam optimizer beta_1 (default: 0.5)
        """
        self.channels = config.get('channels', 1)
        self.lambda_l1 = config.get('lambda_l1', 100)
        self.learning_rate = config.get('learning_rate', 2e-4)
        self.beta_1 = config.get('beta_1', 0.5)
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Loss function
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=self.beta_1
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=self.beta_1
        )
    
    def _downsample(self, filters, size, apply_batchnorm=True):
        """Build downsampling block (encoder)"""
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def _upsample(self, filters, size, apply_dropout=False):
        """Build upsampling block (decoder)"""
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def _build_generator(self):
        """Build U-Net generator"""
        inputs = tf.keras.layers.Input(shape=[256, 256, self.channels])
        
        down_stack = [
            self._downsample(64, 4, apply_batchnorm=False),
            self._downsample(128, 4),
            self._downsample(128, 4),
            self._downsample(256, 4),
            self._downsample(512, 4),
            self._downsample(512, 4),
        ]
        
        up_stack = [
            self._upsample(512, 4, apply_dropout=True),
            self._upsample(512, 4),
            self._upsample(256, 4),
            self._upsample(128, 4),
            self._upsample(128, 4),
            self._upsample(64, 4),
        ]
        
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            self.channels, 4, strides=2, padding='same',
            kernel_initializer=initializer, activation='tanh'
        )
        
        x = inputs
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            # Skip connections commented out as in your original
            # x = tf.keras.layers.Concatenate()([x, skip])
        
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def _build_discriminator(self):
        """Build PatchGAN discriminator"""
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, self.channels], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, self.channels], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])

        down1 = self._downsample(64, 4, False)(x)
        down2 = self._downsample(128, 4)(down1)
        down3 = self._downsample(256, 4)(down2)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def _generator_loss(self, disc_generated_output, gen_output, target):
        """Calculate generator loss (GAN loss + L1 loss)"""
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.lambda_l1 * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def _discriminator_loss(self, disc_real_output, disc_generated_output):
        """Calculate discriminator loss"""
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    # BaseModel interface implementation
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Run generator inference"""
        return self.generator(inputs, training=False)

    @tf.function
    def train_step(self, batch: Tuple[Any, Any]) -> Dict[str, float]:
        """Single training step for both generator and discriminator"""
        input_image, target = batch
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)
            
            # Calculate losses
            gen_total_loss, gen_gan_loss, gen_l1_loss = self._generator_loss(
                disc_generated_output, gen_output, target
            )
            disc_loss = self._discriminator_loss(disc_real_output, disc_generated_output)

        # Calculate gradients
        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        return {
            "gen_total_loss": float(gen_total_loss),
            "gen_gan_loss": float(gen_gan_loss), 
            "gen_l1_loss": float(gen_l1_loss),
            "disc_loss": float(disc_loss),
            "total_loss": float(gen_total_loss + disc_loss)
        }

    def validation_step(self, batch: Tuple[Any, Any]) -> Dict[str, float]:
        """Validation step (no gradient updates)"""
        input_image, target = batch
        
        # Forward pass without training
        gen_output = self.generator(input_image, training=False)
        disc_real_output = self.discriminator([input_image, target], training=False)
        disc_generated_output = self.discriminator([input_image, gen_output], training=False)
        
        # Calculate losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = self._generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = self._discriminator_loss(disc_real_output, disc_generated_output)
        
        return {
            "val_gen_total_loss": float(gen_total_loss),
            "val_gen_gan_loss": float(gen_gan_loss),
            "val_gen_l1_loss": float(gen_l1_loss), 
            "val_disc_loss": float(disc_loss),
            "val_total_loss": float(gen_total_loss + disc_loss)
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Return optimizers for both generator and discriminator"""
        return {
            "generator": self.generator_optimizer,
            "discriminator": self.discriminator_optimizer
        }

    def save(self, path: str) -> None:
        """Save both generator and discriminator"""
        os.makedirs(path, exist_ok=True)
        self.generator.save(os.path.join(path, "generator.keras"))
        self.discriminator.save(os.path.join(path, "discriminator.keras"))
        
        # Save config for reconstruction
        config = {
            "channels": self.channels,
            "lambda_l1": self.lambda_l1,
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1
        }
        import json
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str, **kwargs) -> "Pix2PixGAN":
        """Load model from saved weights and config"""
        import json
        
        # Load config
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(config)
        
        # Load weights
        model.generator = tf.keras.models.load_model(os.path.join(path, "generator.keras"))
        model.discriminator = tf.keras.models.load_model(os.path.join(path, "discriminator.keras"))
        
        return model