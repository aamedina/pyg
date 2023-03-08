(ns net.wikipunk.rdf.torch
  "RDF vocabulary for PyTorch"
  {:rdf/type :owl/Ontology}
  (:require   
   [net.wikipunk.rdf.py]))

(def Module
  "The base class for all neural network modules.

  Your models should also subclass this class.

  Modules can also contain other Modules, allowing to nest them in a tree structure."
  {:db/ident        :torch/Module
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object
   :rdfs/seeAlso    "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module"})

(def DataParallel
  "Implements data parallelism at the module level.

  This container parallelizes the application of the given module by
  splitting the input across the specified devices by chunking in the
  batch dimension (other objects will be copied once per device). In
  the forward pass, the module is replicated on each device, and each
  replica handles a portion of the input. During the backwards pass,
  gradients from each replica are summed into the original module.

  The batch size should be larger than the number of GPUs used."
  {:db/ident        :torch/DataParallel
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html"})

(def Dataset
  "An abstract class representing a Dataset.

  All datasets that represent a map from keys to data samples should
  subclass it. All subclasses should overwrite __getitem__(),
  supporting fetching a data sample for a given key. Subclasses could
  also optionally overwrite __len__(), which is expected to return the
  size of the dataset by many Sampler implementations and the default
  options of DataLoader."
  {:db/ident        :torch/Dataset
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object
   :rdfs/seeAlso    "https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset"})

(def DataLoader
  "Data loader. Combines a dataset and a sampler, and provides an
  iterable over the given dataset.

  The DataLoader supports both map-style and iterable-style datasets
  with single or multi-process loading, customizing loading order and
  optional automatic batching (collation) and memory pinning."
  {:db/ident        :torch/DataLoader
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object
   :rdfs/seeAlso    "https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader"})

(def Sampler
  "Base class for all Samplers."
  {:db/ident        :torch/Sampler
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object
   :rdfs/seeAlso    "https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler"})

(def SequentialSampler
  "Samples elements sequentially, always in the same order."
  {:db/ident        :torch/SequentialSampler
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Sampler})

(def RandomSampler
  "Samples elements randomly. If without replacement, then sample from
  a shuffled dataset. If with replacement, then user can specify
  num_samples to draw."
  {:db/ident        :torch/RandomSampler
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Sampler})

(def SubsetRandomSampler
  "Samples elements randomly from a given list of indices, without replacement."
  {:db/ident        :torch/SubsetRandomSampler
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Sampler})

(def WeightedRandomSampler
  "Samples elements from [0,..,len(weights)-1] with given probabilities (weights)."
  {:db/ident        :torch/WeightedRandomSampler
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Sampler})

(def Tensor
  "A Tensor is a multi-dimensional matrix containing elements of a
  single data type."
  {:db/ident        :torch/Tensor,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :py/Object})
