"""Module routines for training an expansion model"""
import argparse
from typing import Sequence, Optional, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Lambda, Activation, Add
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
from collections import Counter
from tensorflow.keras.layers import Layer

import math
from tensorflow.keras.metrics import top_k_categorical_accuracy
import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import he_normal, Zeros
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.losses import CategoricalCrossentropy

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy

from collections import defaultdict

import itertools
import random
import csv
import time

from aizynthtrain.utils.configs import (
    ExpansionModelPipelineConfig,
    load_config,
)
from aizynthtrain.utils.keras_utils import (
    InMemorySequence,
    setup_callbacks,
    train_keras_model,
    top10_acc,
    top50_acc,
    fine_tune_prototypes,
    train_distillation_model,
    train_stack_model,
    smoothen_labels_sample_weighted_loss,
    fine_tuning_model,
    ProxyAnchor_model_train
)
from aizynthtrain.utils.keras_utils import ProxyAnchorLoss




class ExpansionModelSequence(InMemorySequence):
    """
    Custom sequence class to keep sparse, pre-computed matrices in memory.
    Batches are created dynamically by slicing the in-memory arrays
    The data will be shuffled on each epoch end

    :ivar output_dim: the output size (number of templates)
    """

    def __init__(
        self, input_filename: str, output_filename: str, batch_size: int
    ) -> None:
        super().__init__(input_filename, output_filename, batch_size)
        self.output_dim = self.label_matrix.shape[1]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        idx_ = self._make_slice(idx)
        return self.input_matrix[idx_].toarray(), self.label_matrix[idx_].toarray()

    def on_epoch_end(self) -> None:
        self.input_matrix, self.label_matrix = shuffle(
            self.input_matrix, self.label_matrix, random_state=0
        )




@register_keras_serializable(package='Custom', name='MultiAttention')
class MultiAttention(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads,
        })
        return config


@register_keras_serializable(package='Custom', name='PositionalEmbedding')
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config


def resnet_block(input_layer, units, dropout_rate=0.1):
    x = Dense(units, activation='relu')(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Dense(units)(x)
    x = layers.add([input_layer, x])
    x = layers.Activation('relu')(x)
    return x




####################################################################################
#                         This part is used for class_weights                      #
####################################################################################

def get_class_distribution(sequence):
    num_classes = sequence[0][1].shape[1]
    class_counter = np.zeros(num_classes, dtype=int)

    total_instances = 0
    for batch_inputs, batch_labels in sequence:
        class_counter += np.sum(batch_labels, axis=0)
        total_instances += batch_labels.shape[0]
    class_dist = class_counter / total_instances
    class_dist = class_dist.reshape(1, -1)

    return class_counter, class_dist


# Add weights to samples based on their respective class.
# This adjusts the training process by giving more or less importance to specific classes,
# which can help address class imbalance or emphasize certain categories.

def dataAugmentation(sequence):
    class_counter, class_dist = get_class_distribution(sequence)
    largest_value = np.max(class_counter)
    total = np.sum(class_counter)
    frac = largest_value / total
    mu = math.e * frac
    scores = np.zeros_like(class_counter, dtype=float)

    for i, count in enumerate(class_counter):
        scores[i] = math.log(mu * total / float(count))


    # Check if scores contain inf and replace them with the maximum finite value
    if np.isinf(scores).any():
        max_finite_value = np.max(scores[np.isfinite(scores)])
        scores[np.isinf(scores)] = max_finite_value

    return scores

# Same but return dictionary
def dataAugmentation_dict(sequence):
    class_counter, class_dist = get_class_distribution(sequence)
    largest_value = np.max(class_counter)
    total = np.sum(class_counter)
    frac = largest_value / total
    mu = math.e * frac

    scores = {}

    for i, count in enumerate(class_counter):
        scores[i] = math.log(mu * total / float(count))
    return scores




##########################################################################################################################
#                         This part is used for calculating the predictions based on cosine-similarity                   #
##########################################################################################################################



def calculate_avg_inputs(sequence, model, num_labels):
    # Predict the input shape
    input_shape = model.predict(sequence[0][0]).shape[1:]

    # Initialize sums and counts for each label
    label_sums = np.zeros((num_labels, *input_shape))
    label_counts = np.zeros(num_labels)

    # Accumulate sums and counts
    for inputs, labels in sequence:
        evaluated_inputs = model.predict(inputs)

        for input_, label in zip(evaluated_inputs, labels):
            label_index = np.argmax(label)
            label_sums[label_index] += input_
            label_counts[label_index] += 1

    # Calculate averages, avoiding division by zero
    label_avgs = np.zeros_like(label_sums)
    for i in range(num_labels):
        if label_counts[i] > 0:
            label_avgs[i] = label_sums[i] / label_counts[i]
        else:
            label_avgs[i] = np.zeros_like(label_sums[i])

    return label_avgs


def predict_labels_from_sims(inputs, label_avg_embed, biases=None, top_k=1):
    normed_inputs_encoded = inputs / np.linalg.norm(inputs)
    label_avg_embed = label_avg_embed / np.linalg.norm(label_avg_embed)

    similarities = cosine_similarity(normed_inputs_encoded, label_avg_embed)

    if biases is not None:
        similarities += biases
    # Predict the labels with the highest similarity
    if top_k == 1:
        predicted_labels = np.argmax(similarities, axis=1)
    else:
        predicted_labels = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
    return predicted_labels

def calculate_accuracy(sequence, model, label_avg_embed, result, biases=None):
    correct_predictions = 0
    correct_top10_predictions = 0
    total_predictions = 0

    classes_to_evaluate = [cls for cls, count in enumerate(result) if count < 50]

    for inputs, labels in sequence:
        evaluated_inputs = model.predict(inputs)
        true_labels = np.argmax(labels, axis=1)

        predicted_labels = predict_labels_from_sims(evaluated_inputs, label_avg_embed, biases, top_k=1)
        predicted_top10_labels = predict_labels_from_sims(evaluated_inputs, label_avg_embed, biases, top_k=10)

        for i in range(len(true_labels)):
            if true_labels[i] in classes_to_evaluate:
                total_predictions += 1

                if true_labels[i] == predicted_labels[i]:
                    correct_predictions += 1

                if true_labels[i] in predicted_top10_labels[i]:
                    correct_top10_predictions += 1

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        top10_accuracy = correct_top10_predictions / total_predictions
    else:
        accuracy = 0.0
        top10_accuracy = 0.0

    print(f"Accuracy for classes <50 samples: {accuracy}")
    print(f"Top-10 Accuracy for classes <50 samples: {top10_accuracy}")

    return accuracy, top10_accuracy


def cos_similarity(sequence,sequence_base,labels_base,model):
    top_1_correct = 0
    top_10_correct = 0
    total_samples = 0
    chunk_size = 256
    num_chunks = len(sequence_base) // chunk_size + (1 if len(sequence_base) % chunk_size != 0 else 0)
    for step,(inputs,labels) in enumerate(sequence):
        evaluated_inputs = model.predict(inputs)
        total_samples += len(inputs)
        all_sims = None

        # Calculate cosine similarities in chunks
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, len(sequence_base))
            chunk_sims = cosine_similarity(evaluated_inputs, sequence_base[start:end])

            if all_sims is None:
                all_sims = chunk_sims
            else:
                all_sims = np.concatenate((all_sims, chunk_sims), axis=1)
        #sims = np.array(all_sims)
        sorted_indices = np.argsort(all_sims, axis=1)[:, ::-1]
        for i in range(len(evaluated_inputs)):
            top_10_labels = []
            seen_classes = set()

            for idx in sorted_indices[i]:
                label = labels_base[idx]
                if label not in seen_classes:
                    top_10_labels.append(label)
                    seen_classes.add(label)
                if len(top_10_labels) == 10:
                    break

            true_label = np.argmax(labels[i])
            # Top-1 accuracy
            if true_label == top_10_labels[0]:
                top_1_correct += 1

            # Top-10 accuracy
            if true_label in top_10_labels:
                top_10_correct += 1

        print("ACCURACY FOR THIS STEP: ",top_1_correct / total_samples,"ACCURACY_TOP10 FOR THIS STEP: ", top_10_correct / total_samples)

    top_1_accuracy = top_1_correct / total_samples
    top_10_accuracy = top_10_correct / total_samples

    print(f"Top-1 Accuracy: {top_1_accuracy:.4f}")
    print(f"Top-10 Accuracy: {top_10_accuracy:.4f}")



def l2_norm(input):
    norm = tf.norm(input, ord='euclidean', axis=1, keepdims=True)
    output = input / norm
    return output






##########################################################################################################################
# This part is used for duplicating the minor class in order to have enough interaction between data-data with Proxy Loss#
##########################################################################################################################

def npz_dataset(path):
    npz_data = np.load(path)
    return npz_data

# Return the number of samples for each class
def class_train_dist(npz_labels_data):
    data_indices = npz_labels_data['indices']
    value_counts = {}
    for value in data_indices:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    return value_counts

# Return the sample's position in the train_seq for each class
def class_train_idx(npz_labels_data):
    data_indices = npz_labels_data['indices']
    value_pos = {}

    for idx, value in enumerate(data_indices):
        if value in value_pos:
            value_pos[value].append(idx)
        else:
            value_pos[value] = [idx]

    return value_pos

# data_indptr and data_indices must be in inputs, not in labels
def extract_input(data_indptr, data_indices, position):

    inputs = data_indices[data_indptr[position]:data_indptr[position+1]]

    return inputs


def duplicate_minor_labels(npz_labels_data,k=50):
    data_indices = npz_labels_data['indices']
    data_data = npz_labels_data['data']
    data_shape = npz_labels_data['shape']
    data_indptr = npz_labels_data['indptr']
    data_format = npz_labels_data['format']


    #Duplicate minor classes
    value_counts = {}
    for value in data_indices:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    classes_needing_samples = {value: k - count for value, count in value_counts.items() if count < k}

    samples_to_add = []
    for value, deficit in classes_needing_samples.items():
        samples_to_add.extend([value] * deficit)

    random.shuffle(samples_to_add)
    data_indices = np.append(data_indices, samples_to_add)

    # Difference between new and original length
    length_difference = len(data_indices) - len(npz_labels_data['indices'])

    # Extend the data_data, ie: add 1 to have the same size of len(indices)
    array_to_add = np.ones(length_difference)
    data_data = np.append(data_data, array_to_add)

    # Extend the data_indptr
    array_to_add = np.zeros(length_difference, dtype=int)

    # Calculate each element based on the rule
    for i in range(length_difference):
        array_to_add[i] = data_indptr[len(data_indptr)-1] + 1 + i

    # Append array_to_add to data_indptr
    data_indptr = np.append(data_indptr, array_to_add)

    # Change the shape of data_shape
    data_shape[0] = len(data_indices)

    np.savez('modified_labels_data.npz',
             indices=data_indices,
             indptr=data_indptr,
             format=data_format,
             shape=data_shape,
             data=data_data)


def duplicate_minor_inputs(original_npz_inputs_data, duplicated_minor_labels_npz_data, minor_class_dict):
    origin_length = original_npz_inputs_data['shape'][0]
    duplicated_length = duplicated_minor_labels_npz_data['shape'][0]
    length_difference = duplicated_length - origin_length

    data_indices = original_npz_inputs_data['indices']
    data_data = original_npz_inputs_data['data']
    data_shape = original_npz_inputs_data['shape']
    data_indptr = original_npz_inputs_data['indptr']
    data_format = original_npz_inputs_data['format']


    original_length_data_indices = len(data_indices)
    current_len = len(data_indices)
    print("START")
    for i in range(origin_length,duplicated_length):
        class_to_add = duplicated_minor_labels_npz_data['indices'][i] # Class that duplicated at position i-th
        sample_to_add = random.choice(minor_class_dict[class_to_add]) # Return a random sample (ie: position in original labels) in the class_to_add
        value_to_add = extract_input(data_indptr,data_indices,sample_to_add)
        current_len += len(value_to_add)
        data_indptr = np.append(data_indptr,current_len)
        data_indices = np.append(data_indices,value_to_add)
        if (i - origin_length + 1) % 10000 == 0:
            print(f"Done {(i - origin_length + 1) / length_difference}")

    # Append an array of ones with length = length_difference to data_data
    ones_to_add = np.ones(len(data_indices)-original_length_data_indices)
    data_data = np.append(data_data, ones_to_add)


    if data_indptr[-1] == len(data_indices):
        print("The last element of data_indptr matches the length of data_indices.")
    else:
        print("The last element of data_indptr does not match the length of data_indices.")

    # Update the shape to reflect the new number of samples
    data_shape[0] = duplicated_length





    np.savez('modified_inputs_data.npz',
             indices=data_indices,
             indptr=data_indptr,
             format=data_format,
             shape=data_shape,
             data=data_data)





##########################################################################################################################
#                                           Perform subclass mapping using kmeans                                        #
##########################################################################################################################

def data_infos_dict(sequence):
    class_position_dict = {}
    batch_size = sequence[0][0].shape[0] 

    for batch_index, (batch_inputs, batch_labels) in enumerate(sequence):
        batch_labels_categorical = np.argmax(batch_labels, axis=1)

        for class_idx in np.unique(batch_labels_categorical):
            class_positions = np.where(batch_labels_categorical == class_idx)[0]

            if class_idx not in class_position_dict:
                class_position_dict[class_idx] = []

            class_position_dict[class_idx].extend([batch_index * batch_size + pos for pos in class_positions])

    return class_position_dict


def extract_class_elements(sequence, class_to_extract, data_infos_dict, chunk_size=1000):
    positions = data_infos_dict[class_to_extract]

    def chunk_generator():
        for i in range(0, len(positions), chunk_size):
            chunk_positions = positions[i:i + chunk_size]
            chunk_elements = []
            for position in chunk_positions:
                batch_index = position // sequence[0][0].shape[0]
                within_batch_position = position % sequence[0][0].shape[0]

                batch_inputs, batch_labels = sequence[batch_index]
                chunk_elements.append(batch_inputs[within_batch_position])

            if chunk_elements:
                yield np.array(chunk_elements)
            else:
                yield np.array([])

    return chunk_generator


def compute_centroids(model, sequence, data_infos_dict, chunk_size=1000, output_csv='centroids.csv'):

    with open(output_csv, 'w') as f:
        header_written = False

        for cls, positions in data_infos_dict.items():
            num_samples = len(positions)

            if num_samples >= 2*chunk_size:
                nb_clusters = 6
                kmeans = MiniBatchKMeans(n_clusters=nb_clusters, batch_size=chunk_size, random_state=0)
            else:
                nb_clusters = min(6, 1 + num_samples // 300)
                kmeans = KMeans(n_clusters=nb_clusters, random_state=0)

            class_samples = extract_class_elements(sequence, cls, data_infos_dict)

            if num_samples >= 2*chunk_size:
                for chunk in class_samples():
                    class_samples_encoded = model(chunk, training=False)
                    kmeans.partial_fit(class_samples_encoded)
            else:
                for chunk in class_samples():
                    class_samples_encoded = model(chunk, training=False)
                    kmeans.fit(class_samples_encoded)

            centroids = kmeans.cluster_centers_
            centroids_str = [",".join(map(str, centroid)) for centroid in centroids]

            df = pd.DataFrame({'centroids': centroids_str, 'labels': cls})
            df.to_csv(f, header=f.tell() == 0, index=False)

            if cls % 1000 == 0:
                print(f'Done class: {cls} ')




##########################################################################################################################
#                                           Perform predictions using centroids                                          #
##########################################################################################################################

def read_centroids_from_csv(input_csv: str):
    df = pd.read_csv(input_csv)

    centroids = np.array([list(map(float, centroid.split(','))) for centroid in df['centroids']])

    labels = df['labels'].values

    return centroids, labels


# The main idea is to evaluate the performance of the backbone model 
# without any further fine-tuning.
# This helps assess how well the pre-trained backbone performs on the task 
# before applying additional adjustments or training.

def predict_with_centroids(sequence, model, centroids, data_infos_dict, subclasses_mapped=False):
    total_samples = 0
    correct_top_1 = 0
    correct_top_k = 0

    correct_top_k_minor = 0
    correct_top_k_major = 0
    correct_top_k_large = 0

    nb_samples_minor = 0
    nb_samples_major = 0
    nb_samples_large = 0

    correct_top_k_tiny = 0
    nb_samples_tiny = 0

    for batch_inputs, batch_labels in sequence:
        batch_encoded = model(batch_inputs, training=False)
        similarities = cosine_similarity(batch_encoded, centroids)
        if subclasses_mapped!=False:
            similarities=aggregate_predictions(similarities, subclasses_mapped)
        top_1_classes = np.argmax(similarities, axis=1)
        top_10_classes = np.argsort(-similarities, axis=1)[:, :3]

        batch_size = len(batch_labels)
        total_samples += batch_size

        for i in range(batch_size):
            true_label_index = np.argmax(batch_labels[i])
            if true_label_index == top_1_classes[i]:
                correct_top_1 += 1
            if true_label_index in top_10_classes[i]:
                correct_top_k += 1

            if true_label_index not in data_infos_dict:
                continue

            class_data_info = data_infos_dict[true_label_index]
            num_samples = len(class_data_info)

            if num_samples < 2:
                nb_samples_tiny += 1
                if true_label_index in top_10_classes[i]:
                    correct_top_k_tiny += 1
            elif num_samples < 10:
                nb_samples_minor += 1
                if true_label_index in top_10_classes[i]:
                    correct_top_k_minor += 1
            elif num_samples < 500:
                nb_samples_major += 1
                if true_label_index in top_10_classes[i]:
                    correct_top_k_major += 1
            else:
                nb_samples_large += 1
                if true_label_index in top_10_classes[i]:
                    correct_top_k_large += 1

        current_topk_acc = correct_top_k / total_samples
        current_top1_acc = correct_top_1 / total_samples

        current_topk_acc_major = correct_top_k_major / nb_samples_major if nb_samples_major > 0 else 0
        current_topk_acc_minor = correct_top_k_minor / nb_samples_minor if nb_samples_minor > 0 else 0
        current_topk_acc_tiny = correct_top_k_tiny / nb_samples_tiny if nb_samples_tiny > 0 else 0
        current_topk_acc_large = correct_top_k_large / nb_samples_large if nb_samples_large > 0 else 0

        print(f"Current top1_acc: {current_top1_acc:.4f} with Acc_Top10: {current_topk_acc:.4f}, "
              f"TINY: {current_topk_acc_tiny:.4f}, MINOR: {current_topk_acc_minor:.4f}, "
              f"MAJOR: {current_topk_acc_major:.4f}, LARGE: {current_topk_acc_large:.4f}")

    top_1_accuracy = correct_top_1 / total_samples
    top_k_accuracy = correct_top_k / total_samples
    top_k_accuracy_major = correct_top_k_major / nb_samples_major if nb_samples_major > 0 else 0
    top_k_accuracy_minor = correct_top_k_minor / nb_samples_minor if nb_samples_minor > 0 else 0
    top_k_accuracy_tiny = correct_top_k_tiny / nb_samples_tiny if nb_samples_tiny > 0 else 0

    print(f"Final top1_acc: {top_1_accuracy:.4f}, with Top 10 {top_k_accuracy:.4f}, "
          f"MINOR: {top_k_accuracy_minor:.4f}, MAJOR: {top_k_accuracy_major:.4f}, TINY: {top_k_accuracy_tiny:.4f}")

    return top_1_accuracy, top_k_accuracy



################################################################################################################
#                                           Perform sub-class mapping                                          #
################################################################################################################

def create_subclass_mapping(labels):
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    subclass_mapping = {}
    current_subclass = 0
    for label, count in label_counts.items():

        subclasses = list(range(current_subclass, current_subclass + count))
        subclass_mapping[label] = subclasses
        current_subclass += len(subclasses)

    return subclass_mapping

def generate_subclass(sequence, centroids, subclass_mapped, data_infos, model=None):
    mapped_classes = []
    subclass_counts = defaultdict(int)  # To track the number of samples per subclass

    for step, (x, y) in enumerate(sequence):
        batch_true_classes = np.argmax(y, axis=1)

        if model is not None:
            embedded_inputs = model(x, training=False)
        else:
            embedded_inputs = x

        all_subclasses = []

        for true_class in batch_true_classes:
            all_subclasses.extend(subclass_mapped[true_class])
        all_subclasses = np.unique(all_subclasses)
        centroids_reduced = centroids[all_subclasses]
        cos_sim = cosine_similarity(embedded_inputs, centroids_reduced)

        for sample_idx in range(len(batch_true_classes)):
            true_class = batch_true_classes[sample_idx]
            subclasses_of_true_class = subclass_mapped[true_class]
            subclasses_mask = np.isin(all_subclasses, subclasses_of_true_class)
            cos_sim_relevant = cos_sim[sample_idx, subclasses_mask]

            # Determine max_samples_per_subclass based on the true_class and ensure it's at least 300
            max_samples_per_subclass = max(1+len(data_infos[true_class]) // 4, 300)

            # Sort the subclasses by cosine similarity, in descending order
            sorted_subclasses = np.argsort(cos_sim_relevant)[::-1]

            # Try to find a subclass that has not reached its limit
            selected_subclass = None
            for subclass_idx in sorted_subclasses:
                candidate_subclass = subclasses_of_true_class[subclass_idx]
                if subclass_counts[candidate_subclass] < max_samples_per_subclass:
                    selected_subclass = candidate_subclass
                    break

            # If a suitable subclass was found, add it to mapped_classes
            if selected_subclass is not None:
                mapped_classes.append(selected_subclass)
                subclass_counts[selected_subclass] += 1

        if (step + 1) % 200 == 0:
            print(f"Done {step + 1} batches")

    return mapped_classes

# return an subclassed labels for training set 
def subclasses_to_npz(standard_labels_npz_file_path, subclasses_mapped, updated_labels_npz_file_path='updated_uspto_training_labels.npz'):
    origin_data_labels = np.load(standard_labels_npz_file_path)
    updated_data_labels = {}
    for key in origin_data_labels.keys():
        updated_data_labels[key] = origin_data_labels[key]

    updated_data_labels['indices'] = np.array(subclasses_mapped)
    updated_data_labels['shape'] = np.array([len(subclasses_mapped), np.max(subclasses_mapped)+1])

    for key in updated_data_labels.keys():
        if isinstance(origin_data_labels[key], np.ndarray):
            updated_data_labels[key] = updated_data_labels[key].astype(origin_data_labels[key].dtype)
    np.savez(updated_labels_npz_file_path, **updated_data_labels)


################################################################################################################
#                                           Custom layer to fine-tuning on subclass                            #
################################################################################################################

#Remap subclass back to true class  
from tensorflow.keras.layers import Layer
@register_keras_serializable(package='Custom', name='AggregationLayer')
class AggregationLayer(Layer):
    def __init__(self, subclass_mapping, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.subclass_mapping = subclass_mapping
        self.num_major_classes = len(subclass_mapping)
        self.subclass_indices = self._process_subclass_indices()

    def _process_subclass_indices(self):
        max_length = max(len(subclasses) for subclasses in self.subclass_mapping.values())
        subclass_indices = []
        for subclasses in self.subclass_mapping.values():
            if len(subclasses) < max_length:
                subclasses = subclasses + [subclasses[0]] * (max_length - len(subclasses))
            subclass_indices.append(subclasses)
        return tf.constant(subclass_indices, dtype=tf.int32)

    @tf.function
    def call(self, inputs):
        gathered_inputs = tf.gather(inputs, self.subclass_indices, axis=1)
        major_class_preds = tf.reduce_max(gathered_inputs, axis=2)
        major_class_preds_softmax = tf.nn.softmax(major_class_preds)

        return major_class_preds_softmax

    def get_config(self):

        subclass_mapping = {int(k): list(map(int, v)) for k, v in self.subclass_mapping.items()}
        config = super(AggregationLayer, self).get_config()
        config.update({
            'subclass_mapping': subclass_mapping,
        })
        return config


def aggregate_predictions(inputs, subclass_mapping):

    # Process subclass indices to create a padded tensor
    max_length = max(len(subclasses) for subclasses in subclass_mapping.values())
    subclass_indices = []
    for subclasses in subclass_mapping.values():
        if len(subclasses) < max_length:
            subclasses = subclasses + [subclasses[0]] * (max_length - len(subclasses))
        subclass_indices.append(subclasses)

    subclass_indices = tf.constant(subclass_indices, dtype=tf.int32)

    # Gather inputs according to the subclass indices
    gathered_inputs = tf.gather(inputs, subclass_indices, axis=1)

    # Reduce maximum along the subclass axis to get major class predictions
    major_class_preds = tf.reduce_max(gathered_inputs, axis=2)

    # Apply softmax to get final predictions
    major_class_preds_softmax = tf.nn.softmax(major_class_preds)

    return major_class_preds_softmax



##########################################################################################################################
#                                                             MODEL                                                      #
##########################################################################################################################

def main(args: Optional[Sequence[str]] = None) -> None:
    original_length = 658907 # number of sample in training set

    parser = argparse.ArgumentParser("Tool to training an expansion network policy")
    parser.add_argument("config", help="the filename to a configuration file")
    args = parser.parse_args(args)

    config: ExpansionModelPipelineConfig = load_config(
        args.config, "expansion_model_pipeline"
    )
    train_seq = ExpansionModelSequence(
        config.filename("model_inputs", "training"),
        config.filename("model_labels", "training"),
        config.model_hyperparams.batch_size,
    )

    valid_seq = ExpansionModelSequence(
        config.filename("model_inputs", "validation"),
        config.filename("model_labels", "validation"),
        config.model_hyperparams.batch_size,
    )

    test_seq = ExpansionModelSequence(
        config.filename("model_inputs", "testing"),
        config.filename("model_labels", "testing"),
        config.model_hyperparams.batch_size,
    )



    # Backbone training.

    """
    inputs = Input(shape=(train_seq.input_dim,))


    x = Reshape((4, 512))(inputs)
    x = PositionalEmbedding(4, 512)(x)
    x = BatchNormalization()(x)

    for _ in range(2):
        x1 = MultiAttention(512, 64, 4)(x)
        x1 = Dropout(0.2)(x1)
        x1 = BatchNormalization()(x1)
        x = Add()([x1, x])
    x = layers.GlobalMaxPooling1D()(x)

    x = Dense(units=512,
              kernel_initializer=he_normal(),
              use_bias=True,
              bias_initializer=Zeros(),
              )(x)

    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='Encoder_Output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    print("Model: ")
    model.summary()
    callbacks = setup_callbacks(
        config.filename("training_log"), config.filename("training_checkpoint")
    )

    ProxyAnchor_model_train(
        model,
        train_seq,
        valid_seq,
        config.model_hyperparams.epochs)
    """



    # Perform the subclass processing after the backbone model has been trained.
    """
    model_path = "New_ProxyAnchor_model.hdf5"
    model = load_model(model_path)
    data_infos = data_infos_dict(train_seq)
    compute_centroids(model, train_seq, data_infos)
    centroids, centroids_labels = read_centroids_from_csv('centroids.csv')
    subclasses = create_subclass_mapping(centroids_labels)
    subclasses_generated = generate_subclass(valid_seq,centroids,subclasses,data_infos,model)
    subclasses_to_npz('uspto_validation_labels.npz',subclasses_generated)
    """





    # Perform the prediction using centroids.
    """
    centroids, centroids_labels=read_centroids_from_csv('centroids.csv')
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    #subclasses_mapped = create_subclass_mapping(centroids_labels)

    data_infos = data_infos_dict(train_seq)

    model_path = "New_ProxyAnchor_model.hdf5"
    ProxyAnchor_model = load_model(model_path)
    nb_classes = train_seq[0][1].shape[1]

    # Function to check for NaN values in centroids
    def check_for_nan(centroids):
        if np.isnan(centroids).any():
            print("Centroids contain NaN values.")
        else:
            print("Centroids do not contain any NaN values.")

    # Assuming centroids is the result from calculate_avg_inputs
    centroids = calculate_avg_inputs(train_seq, ProxyAnchor_model, nb_classes)

    # Check for NaN values
    check_for_nan(centroids)
    predict_with_centroids(valid_seq,ProxyAnchor_model,centroids,data_infos)
    """
    

    # Fine-tuning with subclass mapping process.
    """
    centroids, centroids_labels=read_centroids_from_csv('centroids.csv')
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    subclasses_mapped = create_subclass_mapping(centroids_labels)

    class_weight = dataAugmentation(train_seq)
    class_weight = tf.reshape(class_weight, [-1, 1])
    model_path = "New_ProxyAnchor_model.hdf5"
    ProxyAnchor_model = load_model(model_path)
    nb_classes = train_seq[0][1].shape[1]

    for layer in ProxyAnchor_model.layers[:-2]:
        layer.trainable = False

    for layer in ProxyAnchor_model.layers[-2:]:
        layer.trainable = True

    total_subclasses = train_seq[0][1].shape[1]
    x = ProxyAnchor_model.output
    x = Dense(units=total_subclasses,
              #kernel_initializer=tf.keras.initializers.Constant(centroids),
              use_bias=True,
              bias_initializer='zeros',
              activation='softmax',
              name='subclass_output')(x)
    

    outputs = AggregationLayer(subclasses_mapped, name='major_class_output')(x)
    model = Model(inputs=ProxyAnchor_model.input, outputs=outputs)
    model.summary()

    callbacks = setup_callbacks(
        config.filename("training_log"), config.filename("training_checkpoint")
    )

    fine_tuning_model(
        model,
        train_seq,
        valid_seq,
        class_weight,
        config.model_hyperparams.epochs,
        subclasses_mapped
    )

    """


    # Fine-tuning without subclass mapping process.

    """
    model_path = "New_ProxyAnchor_model.hdf5"
    ProxyAnchor_model = load_model(model_path)

    class_weight = dataAugmentation(train_seq)
    class_weight = tf.reshape(class_weight, [-1, 1])
    nb_classes = train_seq[0][1].shape[1]

    for layer in ProxyAnchor_model.layers[:-2]:
        layer.trainable = False

    for layer in ProxyAnchor_model.layers[-2:]:
        layer.trainable = True

    centroids = calculate_avg_inputs(train_seq, ProxyAnchor_model, nb_classes)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    x = ProxyAnchor_model.output
    outputs = Dense(units=train_seq.output_dim,
                    kernel_initializer=tf.keras.initializers.Constant(centroids),
                    use_bias = True,
                    bias_initializer='zeros',
                    activation = 'softmax',
                    name='outputs')(x)
    model = Model(inputs=ProxyAnchor_model.input, outputs=outputs)
    model.summary()
    callbacks = setup_callbacks(
        config.filename("training_log"), config.filename("training_checkpoint")
    )

    fine_tuning_model(
        model,
        train_seq,
        valid_seq,
        class_weight,
        config.model_hyperparams.epochs
    )
    """


    # Evaluation on testing set.

    """
    custom_objects = {'top10_acc': top10_acc,'top50_acc': top50_acc}
    class_counter, _ = get_class_distribution(train_seq)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model_path = "model.hdf5"
    model = load_model(model_path,custom_objects)

    @tf.function
    def valid_step(x_val, y_val):
        preds = model(x_val, training=False)
        loss = loss_fn(y_val, preds)
        return loss, preds

    val_top1_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    val_top5_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5)

    val_top10_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
    val_top50_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=50)

    val_loss_metric = tf.keras.metrics.Mean()
    count=0

    for val_step, (x_val, y_val) in enumerate(test_seq):
        count+=len(x_val)
        x_val = tf.convert_to_tensor(x_val)
        y_val = tf.convert_to_tensor(y_val)
        loss, preds = valid_step(x_val, y_val)

        val_loss_metric.update_state(loss)
        val_top1_accuracy_metric.update_state(y_val, preds)
        val_top5_accuracy_metric.update_state(y_val, preds)

        val_top10_accuracy_metric.update_state(y_val, preds)
        val_top50_accuracy_metric.update_state(y_val, preds)

    val_loss = val_loss_metric.result().numpy()
    val_top1_accuracy = val_top1_accuracy_metric.result().numpy()
    val_top5_accuracy = val_top5_accuracy_metric.result().numpy()

    val_top10_accuracy = val_top10_accuracy_metric.result().numpy()
    val_top50_accuracy = val_top50_accuracy_metric.result().numpy()

    print(
     f"Validation Loss: {val_loss:.4f}, Top-1 Accuracy: {val_top1_accuracy:.4f}, Top-5 Accuracy: {val_top5_accuracy:.4f}, Top-10 Accuracy: {val_top10_accuracy:.4f}, Top-50 Accuracy: {val_top50_accuracy:.4f}")
    print("NUMBER OF SAMPLES: ", count)
    """

if __name__ == "__main__":

    main()
