# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization API."""
import sys
import tensorflow as tf


def _is_colab():
  return "google.colab" in sys.modules


if _is_colab():
  from witwidget.notebook.colab.wit import *  # pylint: disable=wildcard-import,g-import-not-at-top
else:
  from witwidget.notebook.jupyter.wit import *  # pylint: disable=wildcard-import,g-import-not-at-top


class WitConfigBuilder(object):
  """Configuration builder for WitWidget settings."""

  def __init__(self, examples):
    """Constructs the WitConfigBuilder object.

    Args:
      examples: A list of tf.Example or tf.SequenceExample proto objects.
      These are the examples that will be displayed in WIT. If not model to
      infer these examples with is specified through the methods on this class,
      then WIT will display the examples for exploration, but no model inference
      will be performed by the tool.
    """
    self.config = {}
    self.set_examples(examples)
    self.set_model_type('classification')
    self.set_label_vocab([])
  
  def build(self):
    """Returns the configuration set through use of this builder object.

    Used by WitWidget to set the settings on an instance of the What-If Tool.
    """
    return self.config
  
  def store(self, key, value):
    self.config[key] = value

  def set_examples(self, examples):
    """Sets the examples to be displayed in WIT.
    
    Args:
      examples: List of example protos.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('examples', examples)
    if len(examples) > 0:
      self.store('are_sequence_examples',
                 isinstance(examples[0], tf.train.SequenceExample))
    return self
      
  def set_model_type(self, model):
    """Sets the type of the model being used for inference.

    Args:
      model: The model type, such as "classification" or "regression".
      The model type defaults to "classification".

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('model_type', model)
    return self
  
  def set_inference_address(self, address):
    """Sets the inference address for model inference through TF Serving.

    Args:
      address: The address of the served model, including port, such as
      "localhost:8888".

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('inference_address', address)
    return self
    
  def set_model_name(self, name):
    """Sets the model name for model inference through TF Serving.

    Setting a model name is required if inferring through a model hosted by
    TF Serving.

    Args:
      name: The name of the model to be queried through TF Serving at the
      address provided by set_inference_address.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('model_name', name)
    return self
  
  def set_model_version(self, version):
    """Sets the optional model version for model inference through TF Serving.

    Args:
      version: The string version number of the model to be queried through TF
      Serving. This is optional, as TF Serving will use the latest model version
      if none is provided.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('model_version', version)
    return self
      
  def set_model_signature(self, signature):
    """Sets the optional model signature for model inference through TF Serving.

    Args:
      signature: The string signature of the model to be queried through TF
      Serving. This is optional, as TF Serving will use the default model
      signature if none is provided.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('model_signature', signature)
    return self
      
  def set_compare_inference_address(self, address):
    """Sets the inference address for model inference for a second model hosted
    by TF Serving.

    If you wish to compare the results of two models in WIT, use this method
    to setup the details of the second model.

    Args:
      address: The address of the served model, including port, such as
      "localhost:8888".

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('inference_address_2', address)
    return self
      
  def set_compare_model_name(self, name):
    """Sets the model name for a second model hosted by TF Serving.

    If you wish to compare the results of two models in WIT, use this method
    to setup the details of the second model.

    Setting a model name is required if inferring through a model hosted by
    TF Serving.

    Args:
      name: The name of the model to be queried through TF Serving at the
      address provided by set_compare_inference_address.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('model_name_2', name)
    return self
  
  def set_compare_model_version(self, version):
    """Sets the optional model version for a second model hosted by TF Serving.

    If you wish to compare the results of two models in WIT, use this method
    to setup the details of the second model.

    Args:
      version: The string version number of the model to be queried through TF
      Serving. This is optional, as TF Serving will use the latest model version
      if none is provided.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('model_version_2', version)
    return self
      
  def set_compare_model_signature(self, signature):
    """Sets the optional model signature for a second model hosted by TF
    Serving.

    If you wish to compare the results of two models in WIT, use this method
    to setup the details of the second model.

    Args:
      signature: The string signature of the model to be queried through TF
      Serving. This is optional, as TF Serving will use the default model
      signature if none is provided.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('model_signature_2', signature)
    return self
      
  def set_uses_predict_api(self, predict):
    """Indicates that the model uses the Predict API, as opposed to the
    Classification or Regression API.

    If the model doesn't use the standard Classification or Regression APIs
    provided through TF Serving, but instead uses the more flexible Predict API,
    then use this method to indicate that. If this is true, then use the
    set_predict_input_tensor and set_predict_output_tensor methods to indicate
    the names of the tensors that are used as the input and output for the
    models provided in order to perform the appropriate inference request.

    Args:
      predict: True if the model or models use the Predict API.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('uses_predict_api', predict)
    return self
  
  def set_max_classes_to_display(self, max_classes):
    """Sets the maximum number of class results to display for multiclass
    classification models.

    When using WIT with a multiclass model with a large number of possible
    classes, it can be helpful to restrict WIT to only display some smaller
    number of the highest-scoring classes as inference results for any given
    example. This method sets that limit.

    Args:
      max_classes: The maximum number of classes to display for inference
      results for multiclass classification models.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('max_classes', max_classes)
    return self
  
  def set_multi_class(self, multiclass):
    """Sets if the model(s) to query are mutliclass classification models.

    Args:
      multiclass: True if the model or models are multiclass classififcation
      models. Defaults to false.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('multiclass', multiclass)
    return self
      
  def set_predict_input_tensor(self, tensor):
    """Sets the name of the input tensor for models that use the Predict API.

    If using WIT with set_uses_predict_api(True), then call this to specify
    the name of the input tensor of the model or models that accepts the
    example proto for inference.

    Args:
      tensor: The name of the input tensor.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('predict_input_tensor', tensor)
    return self

  def set_predict_output_tensor(self, tensor):
    """Sets the name of the output tensor for models that use the Predict API.

    If using WIT with set_uses_predict_api(True), then call this to specify
    the name of the output tensor of the model or models that returns the
    inference results to be explored by WIT.

    Args:
      tensor: The name of the output tensor.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('predict_output_tensor', tensor)
    return self

  def set_label_vocab(self, vocab):
    """Sets the string value of numeric labels for classification models.

    For classification models, the model returns scores for each class ID
    number (classes 0 and 1 for binary classification models). In order for
    WIT to visually display the results in a more-readable way, you can specify
    string labels for each class ID.

    Args:
      vocab: A list of strings, where the string at each index corresponds to
      the label for that class ID. For example ['<=50K', '>50K'] for the UCI
      census binary classification task.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('label_vocab', vocab)
    return self

  def set_estimator_and_feature_spec(self, estimator, feature_spec):
    """Sets the model for inference as a TF Estimator.

    Instead of using TF Serving to host a model for WIT to query, WIT can
    directly use a TF Estimator object as the model to query. In order to
    accomplish this, a feature_spec must also be provided to parse the
    example protos for input into the estimator.

    Args:
      estimator: The TF Estimator which will be used for model inference.
      feature_spec: The feature_spec object which will be used for exmaple
      parsing.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('estimator_and_spec', {
      'estimator': estimator, 'feature_spec': feature_spec})
    self.set_inference_address('estimator')
    self.set_model_name('estimator')
    return self
  
  def set_compare_estimator_and_feature_spec(self, estimator, feature_spec):
    """Sets a second model for inference as a TF Estimator.

    If you wish to compare the results of two models in WIT, use this method
    to setup the details of the second model.

    Instead of using TF Serving to host a model for WIT to query, WIT can
    directly use a TF Estimator object as the model to query. In order to
    accomplish this, a feature_spec must also be provided to parse the
    example protos for input into the estimator.

    Args:
      estimator: The TF Estimator which will be used for model inference.
      feature_spec: The feature_spec object which will be used for exmaple
      parsing.

    Returns:
      self, in order to enabled method chaining.
    """
    self.store('compare_estimator_and_spec', {
      'estimator': estimator, 'feature_spec': feature_spec})
    self.set_compare_inference_address('estimator')
    self.set_compare_model_name('estimator')
    return self
