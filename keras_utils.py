"""Module containing utility classes and routines used in training of policies"""
import functools
from typing import List, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.config import list_physical_devices
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import register_keras_serializable
from keras.callbacks import Callback
import time
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.losses import CategoricalCrossentropy,SparseCategoricalCrossentropy
#from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay

from tensorflow.keras.callbacks import (
    EarlyStopping,
    CSVLogger,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.metrics import top_k_categorical_accuracy
from scipy import sparse

top10_acc = functools.partial(top_k_categorical_accuracy, k=10)
top10_acc.__name__ = "top10_acc"  # type: ignore

top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
top50_acc.__name__ = "top50_acc"  # type: ignore


@register_keras_serializable(package='Custom', name='Custom_Acc')
class Custom_Accuracy(tf.keras.metrics.Metric):
    def __init__(self,class_weight=None, name='custom_accuracy', **kwargs):
        super(Custom_Accuracy, self).__init__(name=name, **kwargs)
        self.class_weight = class_weight
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.correct_predictions = self.add_weight(name='correct_predictions', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_cast = tf.cast(y_true, dtype=tf.float32)
        y_true_cast = tf.reshape(y_true_cast, tf.shape(y_pred))

        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        correct = tf.cast(tf.equal(tf.cast(y_true, tf.int64), tf.cast(y_pred, tf.int64)), tf.float32)
        num_samples = tf.shape(y_true)[0]
        if self.class_weight is not None:
            reshaped_class_weight = tf.reshape(self.class_weight, [-1, 1])
            reshaped_class_weight = tf.cast(reshaped_class_weight, dtype=tf.float32)
            #weight = tf.matmul(y_true_cast, self.class_weight)
            weight = tf.matmul(y_true_cast, reshaped_class_weight)

            weight = tf.where(weight < 1, 0.0, weight)
            weight = tf.where(weight >= 1, 1.0, weight)
            weight = tf.reshape(weight, [-1])
            #class_weight = tf.cast(class_weight, tf.float32)
            # only interact with samples that have class_weight >=1.0
            correct = tf.reduce_sum(correct * weight , axis = -1)
            num_samples_weighted = tf.reduce_sum(tf.cast(weight > 0.0, tf.int32))
            num_samples = num_samples_weighted

        self.total_samples.assign_add(tf.cast(num_samples, tf.float32))

        self.correct_predictions.assign_add(tf.reduce_sum(correct))

    def result(self):
        return self.correct_predictions / self.total_samples

    def reset_states(self):
        self.total_samples.assign(0)
        self.correct_predictions.assign(0)




@register_keras_serializable(package='Custom', name='Custom_Top10_Acc')
class Custom_Top10_Accuracy(tf.keras.metrics.Metric):
    def __init__(self,class_weight=None, name='custom_top10_accuracy', **kwargs):
        super(Custom_Top10_Accuracy, self).__init__(name=name, **kwargs)
        self.class_weight = class_weight
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.correct_predictions = self.add_weight(name='correct_predictions', initializer='zeros')

    def update_state(self, y_true, y_pred,sample_weight=None):
        y_true_cast = tf.cast(y_true, dtype=tf.float32)
        y_true_cast = tf.reshape(y_true_cast, tf.shape(y_pred))

        y_true = tf.argmax(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int64)
        y_true = tf.reshape(y_true, [-1, 1])

        top_k = tf.math.top_k(y_pred, k=10).indices
        top_k = tf.cast(top_k, tf.int64)

        correct = tf.reduce_any(tf.equal(y_true, top_k), axis=-1)
        correct = tf.cast(correct, tf.float32)

        num_samples = tf.shape(y_true)[0]

        if self.class_weight is not None:
            reshaped_class_weight = tf.reshape(self.class_weight, [-1, 1])
            reshaped_class_weight = tf.cast(reshaped_class_weight, dtype=tf.float32)

            weight = tf.matmul(y_true_cast, reshaped_class_weight)

            weight = tf.where(weight < 1, 0.0, weight)
            weight = tf.where(weight >= 1, 1.0, weight)
            weight = tf.reshape(weight, [-1])
            # only interact with samples that have class_weight >=1.0
            correct = tf.reduce_sum(correct * weight , axis = -1)
            num_samples_weighted = tf.reduce_sum(tf.cast(weight >= 1.0, tf.int32))
            num_samples = num_samples_weighted

        # Increment total_samples by the computed number of samples

        self.total_samples.assign_add(tf.cast(num_samples, tf.float32))

        self.correct_predictions.assign_add(tf.reduce_sum(correct))

    def result(self):
        return self.correct_predictions / self.total_samples

    def reset_states(self):
        self.total_samples.assign(0)
        self.correct_predictions.assign(0)






def l2_norm(input):
    norm = tf.norm(input, ord='euclidean', axis=1, keepdims=True)
    output = input / norm
    return output


# Define Proxy to perform Proxy Anchor Loss during backbone training
@register_keras_serializable(package='Custom', name='ProxyAnchorLoss')
class ProxyAnchorLoss(tf.keras.losses.Loss):
    def __init__(self, nb_classes=42134, sz_embed=512, mrg=0.1, alpha=32, class_weight=None, **kwargs):
        super(ProxyAnchorLoss, self).__init__(**kwargs)
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.class_weight = class_weight
        self.proxies = tf.Variable(
            tf.random.normal([nb_classes, sz_embed]),
            trainable=True,
            name='proxies'
        )
        self.proxies.assign(tf.keras.initializers.HeNormal()(self.proxies.shape))
        # Reshape to an appropriate shape for element_wise multiplication
        if self.class_weight is not None:
            self.class_weight = tf.cast(tf.reshape(self.class_weight, (1, -1)), dtype=tf.float32)
    def call(self, y_true, y_pred):
        P = self.proxies

        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=1)
        P_norm = tf.nn.l2_normalize(P, axis=1)

        cos = tf.matmul(y_pred_norm, P_norm, transpose_b=True)

        P_one_hot = tf.cast(y_true, dtype=tf.float32)
        N_one_hot = 1 - P_one_hot

        pos_exp = tf.exp(-self.alpha * (cos - self.mrg))
        neg_exp = tf.exp(self.alpha * (cos + self.mrg))
        # Apply class weights if provided
        if self.class_weight is not None:
            pos_exp = pos_exp * self.class_weight
            neg_exp = neg_exp * self.class_weight

        num_valid_proxies = tf.math.count_nonzero(tf.reduce_sum(P_one_hot, axis=0))
        num_valid_proxies = tf.maximum(num_valid_proxies, 1)

        P_sim_sum = tf.reduce_sum(pos_exp * P_one_hot, axis=0)
        N_sim_sum = tf.reduce_sum(neg_exp * N_one_hot, axis=0)

        pos_term = tf.reduce_sum(tf.math.log(1 + P_sim_sum)) / tf.cast(num_valid_proxies, tf.float32)
        neg_term = tf.reduce_sum(tf.math.log(1 + N_sim_sum)) / tf.cast(self.nb_classes, tf.float32)

        loss = pos_term + neg_term
        return loss

    def get_trainable_vars(self):
        return [self.proxies]


@register_keras_serializable(package='Custom', name='CosineSimilarity')
def cosine_similarity_loss(y_true, y_pred):
    y_true_normalized = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=-1)

    cosine_similarity = tf.reduce_sum(y_true_normalized * y_pred_normalized, axis=-1)

    loss = -cosine_similarity

    return loss



@register_keras_serializable(package='Custom', name='smoothen_labels_sample_weighted_loss')
def smoothen_labels_sample_weighted_loss(class_weight, nb_classes=42134, label_smoothing=0.15):
    def smooth_labels(labels, factor, nb_classes):
        labels = tf.cast(labels, tf.float32)
        labels *= (1.0 - factor)
        labels += (factor / nb_classes)
        return labels

    def custom_loss(y_true, y_pred):
        y_true_cast = tf.cast(y_true, dtype=y_pred.dtype)
        y_true_cast = tf.reshape(y_true_cast, tf.shape(y_pred))

        class_weight_cast = tf.cast(class_weight, dtype=y_pred.dtype)
        sample_weight = tf.matmul(y_true_cast, class_weight_cast)

        y_true_smoothed = smooth_labels(y_true, label_smoothing, nb_classes)

        cce = CategoricalCrossentropy()
        y_true_smoothed = tf.cast(y_true_smoothed, dtype=y_pred.dtype)

        loss = cce(y_true_smoothed, y_pred, sample_weight=sample_weight)

        return loss

    return custom_loss

# AiZynthFinder base model training
class InMemorySequence(Sequence):  # pylint: disable=W0223
    """
    Class for in-memory data management

    :param input_filname: the path to the model input data
    :param output_filename: the path to the model output data
    :param batch_size: the size of the batches
    """

    def __init__(
        self, input_filename: str, output_filename: str, batch_size: int
    ) -> None:
        self.batch_size = batch_size
        self.input_matrix = self._load_data(input_filename)
        self.label_matrix = self._load_data(output_filename)
        self.input_dim = self.input_matrix.shape[1]

    def __len__(self) -> int:
        return int(np.ceil(self.label_matrix.shape[0] / float(self.batch_size)))

    def _make_slice(self, idx: int) -> slice:
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return slice(start, end)

    @staticmethod
    def _load_data(filename: str) -> np.ndarray:
        try:
            return sparse.load_npz(filename)
        except ValueError:
            return np.load(filename)["arr_0"]





def setup_callbacks(
    log_filename: str, checkpoint_filename: str
) -> Tuple[EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau]:


    #early_stopping = EarlyStopping(monitor="val_custom_top10_accuracy", patience=10)
    early_stopping = EarlyStopping(monitor="val_top10_acc", patience=10)
    csv_logger = CSVLogger(log_filename)
    checkpoint = ModelCheckpoint(
        checkpoint_filename,
        monitor='val_top10_acc',
        save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_top10_acc',
        factor=0.5,
        patience=2,
        verbose=0,
        mode="auto",
        min_delta=0.001,
        cooldown=0,
        min_lr=0,
    )

    return [early_stopping, csv_logger, checkpoint, reduce_lr]




def train_keras_model(
    model: Model,
    train_seq: InMemorySequence,
    valid_seq: InMemorySequence,
    class_weight_dict,
    loss: str,
    metrics: List[Any],
    callbacks: List[Any],
    epochs: int,
) -> None:

    print(f"Available GPUs: {list_physical_devices('GPU')}")
    adam = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=adam,
        loss=loss,
        metrics=metrics,
    )

    
    model.fit(
        train_seq,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=valid_seq,
        max_queue_size=20,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        class_weight=class_weight_dict
    )







# Metric Transformation Training
# Training the backbone model using Proxy-based Deep Metric Learning.
# The training loop is customized to incorporate proxies for better representation learning, 
# adapting the model to handle metric learning tasks more effectively.

def ProxyAnchor_model_train(
        model: Model,
        train_seq: tf.keras.utils.Sequence,
        valid_seq: tf.keras.utils.Sequence,
        epochs: int,
) -> np.ndarray:

    print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

    backbone_initial_lr = 0.0005
    proxy_initial_lr = 0.001

    backbone_optimizer = Adam(learning_rate=backbone_initial_lr, beta_1=0.9, beta_2=0.999)
    proxy_optimizer = Adam(learning_rate=proxy_initial_lr, beta_1=0.9, beta_2=0.999)
    nb_classes = train_seq[0][1].shape[1]

    print(f"Number of classes: {nb_classes}")
    proxy_loss = ProxyAnchorLoss(nb_classes=nb_classes,sz_embed=512)

    model.compile(
        loss=proxy_loss
    )

    proxy_vars = proxy_loss.get_trainable_vars()

    total_params = model.count_params()
    print(f"Total Parameters: {total_params}")
    trainable_vars = model.trainable_variables
    total_trainable_params = sum(tf.size(param).numpy() for param in trainable_vars)
    print(f"Total Trainable Parameters: {total_trainable_params}")

    best_val_loss = float('inf')
    patience = 2
    min_loss_reduction = 0.005
    count = 0
    extra_epoch = 5
    min_lr = 0.00005
    epochs_no_improve = 0

    @tf.function
    def train_step(x, y):
        with tf.GradientTape(persistent=True) as tape:
            preds = model(x, training=True)
            loss = proxy_loss(y, preds)
        gradients = tape.gradient(loss, model.trainable_variables)
        proxy_gradients = tape.gradient(loss, proxy_vars)

        backbone_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        proxy_optimizer.apply_gradients(zip(proxy_gradients, proxy_vars))
        return loss
    @tf.function
    def valid_step(x_val, y_val):
        preds = model(x_val, training=False)
        loss = proxy_loss(y_val, preds)
        return loss

    for epoch in range(epochs):
        train_seq.on_epoch_end()
        total_steps = len(train_seq)
        print(f"Epoch {epoch + 1}/{epochs}")
        loss_metric = tf.keras.metrics.Mean()
        for step, (x, y) in enumerate(train_seq):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.int32)
            loss = train_step(x, y)
            loss_metric.update_state(loss)
            if (step+1)%1000==0:
                print(f"Done {step+1}/{total_steps}, Training Loss: {loss_metric.result().numpy():.4f}")
        print(f"Epoch {epoch + 1}/{epochs},  Average Training Loss: {loss_metric.result().numpy():.4f}")

        val_loss_metric = tf.keras.metrics.Mean()
        for val_step, (x_val, y_val) in enumerate(valid_seq):
            x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
            loss = valid_step(x_val, y_val)
            val_loss_metric.update_state(loss)
        val_loss = val_loss_metric.result().numpy()
        print(f"Validation Loss: {val_loss:.4f}")

        if best_val_loss > val_loss + min_loss_reduction:
            best_val_loss = val_loss
            model.save('./New_ProxyAnchor_model.hdf5')
            print(f"Saved model with validation loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and backbone_optimizer.learning_rate.numpy() > min_lr:
            new_lr_backbone = max(0.00005, backbone_optimizer.learning_rate.numpy() * 0.5)
            backbone_optimizer.learning_rate.assign(new_lr_backbone)
            new_lr_proxy = max(0.00005, proxy_optimizer.learning_rate.numpy() * 0.5)
            proxy_optimizer.learning_rate.assign(new_lr_proxy)
            print(f'Current lr_rate: Backbone: {backbone_optimizer.learning_rate.numpy()}, Proxies: {proxy_optimizer.learning_rate.numpy()}')
            epochs_no_improve = 0
        if proxy_optimizer.learning_rate.numpy() <= min_lr:
            if count < extra_epoch:
                count += 1
            else:
                break
    print("Training completed.")



from tensorflow.keras import regularizers
def fine_tuning_model(
        model: Model,
        train_seq: tf.keras.utils.Sequence,
        valid_seq: tf.keras.utils.Sequence,
        class_weight,
        epochs: int,
        subclass_mapped,
) -> np.ndarray:

    print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    total_params = model.count_params()
    print(f"Total Parameters: {total_params}")
    trainable_vars = model.trainable_variables
    total_trainable_params = sum(tf.size(param) for param in trainable_vars)
    print(f"Total Trainable Parameters: {total_trainable_params.numpy()}")

    class_weight = np.array(class_weight, dtype=np.float32)
    class_weight = tf.constant(class_weight, dtype=tf.float32)

    class_weight_dtype = class_weight.dtype

    min_lr = 0.00005
    patience = 2
    lr_factor = 0.5
    no_improvement_threshold = 0.00005

    extra_epoch = 10

    best_val_top10_acc = 0
    epochs_no_improve = 0
    count = 0
    label_smoothing_factor = 0.15

    num_classes = train_seq[0][1].shape[1]
    reverse_mapped = reverse_mapping(subclass_mapped)

    @tf.function
    def train_step(x, y, sample_weight):
        with tf.GradientTape() as tape:
            intermediate_output = truncated_model(x, training=True)
            loss = loss_fn(y, intermediate_output,sample_weight=sample_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, intermediate_output

    @tf.function
    def valid_step(x_val, y_val):
        preds = model(x_val, training=False)
        loss = loss_fn(y_val, preds)
        return loss, preds

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_seq.on_epoch_end()
        loss_metric = tf.keras.metrics.Mean()
        top1_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
        top10_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=10)

        for step, (x, y) in enumerate(train_seq):
            y = tf.cast(y, class_weight.dtype)


            y_smoothen = y * (1 - label_smoothing_factor) + (label_smoothing_factor / num_classes)
            y_smoothen = tf.convert_to_tensor(y_smoothen, dtype=class_weight_dtype)
            sample_weight = tf.matmul(y, class_weight)
            sample_weight = tf.convert_to_tensor(sample_weight)
            x = tf.convert_to_tensor(x)
            # The structure of `truncated_model` depends on whether subclass mapping is applied:
            # - If subclass mapping is used, the model is trained on subclasses instead of true classes.
            # - If subclass mapping is not used, `truncated_model` is identical to the original model 
            #   (i.e., there is no additional layer to remap subclasses back to true classes).
            truncated_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            #truncated_model = model
            loss, intermediate_output = train_step(x, y_smoothen, sample_weight)
            if (epoch+1) % 10 == 0:
                loss_metric.update_state(loss)
                top1_accuracy_metric.update_state(y, intermediate_output)
                top10_accuracy_metric.update_state(y, intermediate_output)

        if (epoch+1) % 10 == 0:
            top1_acc_train = top1_accuracy_metric.result().numpy()
            top10_acc_train = top10_accuracy_metric.result().numpy()
            avg_train_loss = loss_metric.result().numpy()
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f", Average Training Loss: {avg_train_loss:.4f}, "
                  f"Top1_acc: {top1_acc_train:.4f}, "
                  f"Top10_acc: {top10_acc_train:.4f}")
        else:

            print(f"Done Epoch: {epoch+1}/{epochs}")




        val_top1_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
        val_top10_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
        val_loss_metric = tf.keras.metrics.Mean()

        for val_step, (x_val, y_val) in enumerate(valid_seq):
            x_val = tf.convert_to_tensor(x_val)
            y_val = tf.convert_to_tensor(y_val)
            loss, preds = valid_step(x_val, y_val)
            val_loss_metric.update_state(loss)
            val_top1_accuracy_metric.update_state(y_val, preds)
            val_top10_accuracy_metric.update_state(y_val, preds)

        val_loss = val_loss_metric.result().numpy()
        val_top1_accuracy = val_top1_accuracy_metric.result().numpy()
        val_top10_accuracy = val_top10_accuracy_metric.result().numpy()

        print(f"Validation Loss: {val_loss:.4f}, Top-1 Accuracy: {val_top1_accuracy:.4f}, Top-10 Accuracy: {val_top10_accuracy:.4f}")

        if val_top10_accuracy > best_val_top10_acc + no_improvement_threshold:
            best_val_top10_acc = val_top10_accuracy
            model.save('./result.hdf5')
            print(f"Saved model with validation Top-10 accuracy: {val_top10_accuracy:.4f}")
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and optimizer.learning_rate.numpy() > min_lr:
            new_lr = max(optimizer.learning_rate.numpy() * lr_factor, min_lr)
            optimizer.learning_rate.assign(new_lr)
            print(f"Reduced learning rate to {new_lr:.6f}")
            epochs_no_improve = 0

        if optimizer.learning_rate.numpy() <= min_lr:
            if count < extra_epoch:
                count += 1
            else:
                break


    print("Training completed.")











