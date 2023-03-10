(ns net.wikipunk.rdf.pyg
  "RDF vocabulary for PyTorch Geometric"
  {:rdf/type :owl/Ontology}
  (:require
   [net.wikipunk.rdf.py]
   [net.wikipunk.rdf.torch]))

(def MessagePassing
  "Base class for creating message passing layers"
  {:db/ident        :pyg/MessagePassing,
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing"})

(def CuGraphModule
  "An abstract base class for implementing cugraph message passing layers."
  {:db/ident        :pyg/CuGraphModule
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Module})

(def Sequential
  "An extension of the torch.nn.Sequential container in order to
  define a sequential GNN model. Since GNN operators take in multiple
  input arguments, torch_geometric.nn.Sequential expects both global
  input arguments, and function header definitions of individual
  operators. If omitted, an intermediate module will operate on the
  output of its preceding module."
  {:db/ident        :pyg/Sequential,
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.sequential.Sequential"})

(def Linear
  "Applies a linear tranformation to the incoming data"
  {:db/ident        :pyg/Linear,
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.Linear"})

(def HeteroLinear
  "Applies separate linear tranformations to the incoming data according to types"
  {:db/ident        :pyg/HeteroLinear,
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.HeteroLinear"})

;; torch_geometric.data

(def BaseData
  "Abstract base class for PyG data."
  {:db/ident        :pyg/BaseData
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def FeatureStore
  "An abstract base class to access features from a remote feature
  store."
  {:db/ident        :pyg/FeatureStore
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def GraphStore
  "An abstract base class to access edges from a remote graph store."
  {:db/ident        :pyg/GraphStore
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def Data
  "A data object describing a homogeneous graph. The data object can
  hold node-level, link-level and graph-level attributes. In general,
  Data tries to mimic the behavior of a regular Python dictionary. In
  addition, it provides useful functionality for analyzing graph
  structures, and provides basic PyTorch tensor functionalities."
  {:db/ident        :pyg/Data
   :rdf/type        :owl/Class
   :rdfs/subClassOf [:pyg/BaseData :pyg/FeatureStore :pyg/GraphStore]
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs"})

(def HeteroData
  "A data object describing a heterogeneous graph, holding multiple
  node and/or edge types in disjunct storage objects. Storage objects
  can hold either node-level, link-level or graph-level attributes. In
  general, HeteroData tries to mimic the behavior of a regular nested
  Python dictionary. In addition, it provides useful functionality for
  analyzing graph structures, and provides basic PyTorch tensor
  functionalities."
  {:db/ident        :pyg/HeteroData
   :rdf/type        :owl/Class
   :rdfs/subClassOf [:pyg/BaseData :pyg/FeatureStore :pyg/GraphStore]
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData"})

(def Batch
  "A data object describing a batch of graphs as one big
  (disconnected) graph. Inherits from :pyg/Data or :pyg/HeteroData. In
  addition, single graphs can be identified via the assignment vector
  batch, which maps each node to its respective graph identifier."
  {:db/ident        :pyg/Batch
   :rdf/type        :owl/Class
   :rdfs/subClassOf {:owl/unionOf [:pyg/Data :pyg/HeteroData]
                     :rdf/type    :owl/Class}
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html#torch_geometric.data.Batch"})

(def TemporalData
  "A data object composed by a stream of events describing a temporal
  graph. The TemporalData object can hold a list of events (that can
  be understood as temporal edges in a graph) with structured
  messages. An event is composed by a source node, a destination node,
  a timestamp and a message. Any Continuous-Time Dynamic Graph (CTDG)
  can be represented with these four values.

  In general, TemporalData tries to mimic the behavior of a regular
  Python dictionary. In addition, it provides useful functionality for
  analyzing graph structures, and provides basic PyTorch tensor
  functionalities."
  {:db/ident        :pyg/TemporalData
   :rdf/type        :owl/Class
   :rdfs/subClassOf :pyg/BaseData
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.TemporalData.html#torch_geometric.data.TemporalData"})

(def Dataset
  "Dataset base class for creating graph datasets."
  {:db/ident        :pyg/Dataset,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Dataset
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html"})

(def InMemoryDataset
  "Dataset base class for creating graph datasets which easily fit
  into CPU memory."
  {:db/ident        :pyg/InMemoryDataset,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Dataset
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html#creating-in-memory-datasets"})

(def CastMixin
  "Abstract base class for mixins."
  {:db/ident        :pyg/CastMixin
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def TensorAttr
  "Defines the attributes of a FeatureStore tensor. It holds all the
  parameters necessary to uniquely identify a tensor from the
  FeatureStore."
  {:db/ident        :pyg/TensorAttr
   :rdf/type        :owl/Class
   :rdfs/subClassOf :pyg/CastMixin
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.TensorAttr.html#torch_geometric.data.TensorAttr"})

(def EdgeAttr
  "Defines the attributes of a GraphStore edge. It holds all the
  parameters necessary to uniquely identify an edge from the
  GraphStore.

  Note that the order of the attributes is important; this is the
  order in which attributes must be provided for indexing
  calls. GraphStore implementations can define a different ordering by
  overriding EdgeAttr.__init__()."
  {:db/ident        :pyg/EdgeAttr
   :rdf/type        :owl/Class
   :rdfs/subClassOf :pyg/CastMixin
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.EdgeAttr.html#torch_geometric.data.EdgeAttr"})

;; torch_geometric.loader

(def DataLoader
  "A data loader which merges data objects from a
  :pyg/Dataset to a mini-batch. Data objects can be
  either of type :pyg/Data or :pyg/HeteroData."
  {:db/ident        :pyg/DataLoader
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/DataLoader
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.DataLoader"})

(def ClusterData
  "Clusters/partitions a graph data object into multiple subgraphs, as
  motivated by the `\"Cluster-GCN: An Efficient Algorithm for Training
  Deep and Large Graph Convolutional Networks\" paper."
  {:db/ident        :pyg/ClusterData,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Dataset
   :rdfs/seeAlso    "https://arxiv.org/abs/1905.07953"})

(def ClusterLoader
  "The data loader scheme from the `\"Cluster-GCN: An Efficient
  Algorithm for Training Deep and Large Graph Convolutional Networks\"
  paper which merges partioned subgraphs and their between-cluster
  links from a large-scale graph data object to form a mini-batch."
  {:db/ident        :pyg/ClusterLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader
   :rdfs/seeAlso    "https://arxiv.org/abs/1905.07953"})

(def DataListLoader
  "A data loader which batches data objects from a :pyg/Dataset to a
  Python list.  Data objects can be either of type :`:pyg/Data`
  or `:pyg/HeteroData`."
  {:db/ident        :pyg/DataListLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader})

(def DenseDataLoader
  "A data loader which batches data objects from a `:pyg/Dataset` to a
  `:pyg/Batch` object by stacking all attributes in a new dimension."
  {:db/ident        :pyg/DenseDataLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader})

(def DynamicBatchSampler
  "Dynamically adds samples to a mini-batch up to a maximum size
  (either based on number of nodes or number of edges). When data
  samples have a wide range in sizes, specifying a mini-batch size in
  terms of number of samples is not ideal and can cause CUDA OOM
  errors."
  {:db/ident        :pyg/DynamicBatchSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Sampler})

(def GraphSAINTEdgeSampler
  "The GraphSAINT edge sampler class (see
  class:`:pyg/GraphSAINTSampler`)."
  {:db/ident        :pyg/GraphSAINTEdgeSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/GraphSAINTSampler})

(def GraphSAINTNodeSampler
  "The GraphSAINT node sampler class (see
  class:`:pyg/GraphSAINTSampler`)."
  {:db/ident        :pyg/GraphSAINTNodeSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/GraphSAINTSampler})

(def GraphSAINTRandomWalkSampler
  "The GraphSAINT random walk sampler class (see
  class:`:pyg/GraphSAINTSampler`)."
  {:db/ident        :pyg/GraphSAINTRandomWalkSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/GraphSAINTSampler})

(def GraphSAINTSampler
  "The GraphSAINT sampler base class from the `\"GraphSAINT: Graph
  Sampling Based Inductive Learning Method\" paper.  Given a graph in
  a `data` object, this class samples nodes and constructs subgraphs
  that can be processed in a mini-batch fashion.  Normalization
  coefficients for each mini-batch are given via `node_norm` and
  `edge_norm` data attributes."
  {:db/ident        :pyg/GraphSAINTSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader
   :rdfs/seeAlso    "https://arxiv.org/abs/1907.04931"})

(def HGTLoader
  "The Heterogeneous Graph Sampler from the `\"Heterogeneous Graph
  Transformer\" paper.  This loader allows for mini-batch training of
  GNNs on large-scale graphs where full-batch training is not
  feasible."
  {:db/ident        :pyg/HGTLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/NodeLoader
   :rdfs/seeAlso    "https://arxiv.org/abs/2003.01332"})

(def ImbalancedSampler
  "A weighted random sampler that randomly samples elements according
  to class distribution.  As such, it will either remove samples from
  the majority class (under-sampling) or add more examples from the
  minority class (over-sampling)."
  {:db/ident        :pyg/ImbalancedSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/WeightedRandomSampler})

(def LinkLoader
  "A data loader that performs neighbor sampling from link
  information, using a generic class:`:pyg/BaseSampler` implementation
  that defines a `sample_from_edges` function and is supported on the
  provided input `data` object."
  {:db/ident        :pyg/LinkLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader})

(def LinkNeighborLoader
  "A link-based data loader derived as an extension of the node-based
  class:`:pyg/NeighborLoader`.  This loader allows for mini-batch
  training of GNNs on large-scale graphs where full-batch training is
  not feasible."
  {:db/ident        :pyg/LinkNeighborLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/LinkLoader})

(def NeighborLoader
  "A data loader that performs neighbor sampling as introduced in the
  `\"Inductive Representation Learning on Large Graphs\" paper.  This
  loader allows for mini-batch training of GNNs on large-scale graphs
  where full-batch training is not feasible."
  {:db/ident        :pyg/NeighborLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/NodeLoader
   :rdfs/seeAlso    "https://arxiv.org/abs/1706.02216"})

(def NeighborSampler
  "The neighbor sampler from the `\"Inductive Representation Learning
  on Large Graphs\" paper, which allows for mini-batch training of
  GNNs on large-scale graphs where full-batch training is not
  feasible."
  {:db/ident        :pyg/NeighborSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader
   :rdfs/seeAlso    "https://arxiv.org/abs/1706.02216"})

(def NodeLoader
  "A data loader that performs neighbor sampling from node
  information, using a generic class:`:pyg/BaseSampler` implementation
  that defines a `sample_from_nodes` function and is supported on the
  provided input `data` object."
  {:db/ident        :pyg/NodeLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader})

(def RandomNodeLoader
  "A data loader that randomly samples nodes within a graph and
  returns their induced subgraph."
  {:db/ident        :pyg/RandomNodeLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader})

(def ShaDowKHopSampler
  "The ShaDow :math:`k`-hop sampler from the `\"Decoupling the Depth
  and Scope of Graph Neural Networks\" paper.  Given a graph in a
  `data` object, the sampler will create shallow, localized
  subgraphs. A deep GNN on this local graph then smooths the
  informative local signals."
  {:db/ident        :pyg/ShaDowKHopSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader
   :rdfs/seeAlso    "https://arxiv.org/abs/2201.07858"})

(def TemporalDataLoader
  "A data loader which merges succesive events of a
  class:`:pyg/TemporalData` to a mini-batch."
  {:db/ident        :pyg/TemporalDataLoader,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DataLoader})

;; torch_geometric.sampler

(def BaseSampler
  "A base class that initializes a graph sampler and provides
  `sample_from_nodes` and`sample_from_edges` routines."
  {:db/ident        :pyg/BaseSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :py/Object})

(def HGTSampler
  "An implementation of an in-memory heterogeneous layer-wise sampler
  user by `:pyg/HGTLoader`."
  {:db/ident        :pyg/HGTSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseSampler})

(def NeighborSampler
  "An implementation of an in-memory (heterogeneous) neighbor sampler
  used by `:pyg/NeighborLoader`."
  {:db/ident        :pyg/NeighborSampler,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseSampler})

;; torch_geometric.nn

;; Convolutional Layers

(def SimpleConv
  "A simple message passing operator that performs (non-trainable) propagation"
  {:db/ident        :pyg/SimpleConv
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SimpleConv.html#torch_geometric.nn.conv.SimpleConv"})

(def AGNNConv
  "The graph attentional propagation layer from the \"Attention-based
  Graph Neural Network for Semi-Supervised Learning\" paper."
  {:db/ident        :pyg/AGNNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1803.03735"})

(def APPNP
  "The approximate personalized propagation of neural predictions
  layer from the `\"Predict then Propagate: Graph Neural Networks meet
  Personalized PageRank\" paper."
  {:db/ident        :pyg/APPNP,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1810.05997"})

(def ARMAConv
  "The ARMA graph convolutional operator from the `\"Graph Neural
  Networks with Convolutional ARMA Filters\" paper."
  {:db/ident        :pyg/ARMAConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1901.01343"})

(def CGConv
  "The crystal graph convolutional operator from the \"Crystal Graph
  Convolutional Neural Networks for an Accurate and Interpretable
  Prediction of Material Properties\" paper."
  {:db/ident        :pyg/CGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301"})

(def ChebConv
  "The chebyshev spectral graph convolutional operator from the
  \"Convolutional Neural Networks on Graphs with Fast Localized
  Spectral Filtering\" paper."
  {:db/ident        :pyg/ChebConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1606.09375"})

(def ClusterGCNConv
  "The ClusterGCN graph convolutional operator from the \"Cluster-GCN:
  An Efficient Algorithm for Training Deep and Large Graph
  Convolutional Networks\" paper."
  {:db/ident        :pyg/ClusterGCNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1905.07953"})

(def DNAConv
  "The dynamic neighborhood aggregation operator from the `\"Just
  Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural
  Networks\" paper."
  {:db/ident        :pyg/DNAConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1904.04849"})

(def DynamicEdgeConv
  "The dynamic edge convolutional operator from the `\"Dynamic Graph
  CNN for Learning on Point Clouds\" paper (see `:pyg/EdgeConv`),
  where the graph is dynamically constructed using nearest neighbors
  in the feature space."
  {:db/ident        :pyg/DynamicEdgeConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1801.07829"})

(def ECConv
  "The continuous kernel-based convolutional operator from the
  \"Neural Message Passing for Quantum Chemistry\" paper. This
  convolution is also known as the edge-conditioned convolution from
  the \"Dynamic Edge-Conditioned Filters in Convolutional Neural
  Networks on Graphs\" paper."
  {:db/ident        :pyg/ECConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    ["https://arxiv.org/abs/1704.01212"
                     "https://arxiv.org/abs/1704.02901"]})

(def EGConv
  "The Efficient Graph Convolution from the `\"Adaptive Filters and
  Aggregator Fusion for Efficient Graph Convolutions\" paper."
  {:db/ident        :pyg/EGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2104.01481"})

(def EdgeConv
  "The edge convolutional operator from the `\"Dynamic Graph CNN for
  Learning on Point Clouds\" paper"
  {:db/ident        :pyg/EdgeConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1801.07829"})

(def FAConv
  "The Frequency Adaptive Graph Convolution operator from the \"Beyond
  Low-Frequency Information in Graph Convolutional Networks\" paper."
  {:db/ident        :pyg/FAConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2101.00797"})

(def FastRGCNConv
  "See class `:pyg/RGCNConv`."
  {:db/ident        :pyg/FastRGCNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/RGCNConv})

(def FeaStConv
  "The (translation-invariant) feature-steered convolutional operator
  from the `\"FeaStNet: Feature-Steered Graph Convolutions for 3D
  Shape Analysis\" paper."
  {:db/ident        :pyg/FeaStConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1706.05206"})

(def FiLMConv
  "The FiLM graph convolutional operator from the \"GNN-FiLM: Graph
  Neural Networks with Feature-wise Linear Modulation\" paper."
  {:db/ident        :pyg/FiLMConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1906.12192"})

(def FusedGATConv
  "The fused graph attention operator from the \"Understanding GNN
  Computational Graph: A Coordinated Computation, IO, and Memory
  Perspective\" paper."
  {:db/ident        :pyg/FusedGATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/GATConv
   :rdfs/seeAlso    "https://proceedings.mlsys.org/paper/2022/file/9a1158154dfa42caddbd0694a4e9bdc8-Paper.pdf"})

(def GATConv
  "The graph attentional operator from the `\"Graph Attention Networks\" paper."
  {:db/ident        :pyg/GATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1710.10903"})

(def GATv2Conv
  "The GATv2 operator from the \"How Attentive are Graph Attention
  Networks?\" paper, which fixes the static attention problem of the
  standard class:`:pyg/GATConv` layer. Since the linear layers in the
  standard GAT are applied right after each other, the ranking of
  attended nodes is unconditioned on the query node. In contrast, in
  class:`:pyg/GATv2Conv`, every node can attend to any other node."
  {:db/ident        :pyg/GATv2Conv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2105.14491"})

(def GCN2Conv
  "The graph convolutional operator with initial residual connections
  and identity mapping (GCNII) from the \"Simple and Deep Graph
  Convolutional Networks\" paper."
  {:db/ident        :pyg/GCN2Conv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2007.02133"})

(def GCNConv
  "The graph convolutional operator from the `\"Semi-supervised
  Classification with Graph Convolutional Networks\" paper."
  {:db/ident        :pyg/GCNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1609.02907"})

(def GENConv
  "The GENeralized Graph Convolution (GENConv) from the \"DeeperGCN:
  All You Need to Train Deeper GCNs\" paper."
  {:db/ident        :pyg/GENConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2006.07739"})

(def GINConv
  "The graph isomorphism operator from the `\"How Powerful are Graph
  Neural Networks?\" paper."
  {:db/ident        :pyg/GINConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1810.00826"})

(def GINEConv
  "The modified class:`:pyg/GINConv` operator from the `\"Strategies
  for Pre-training Graph Neural Networks\" paper."
  {:db/ident        :pyg/GINEConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1905.12265"})

(def GMMConv
  "The gaussian mixture model convolutional operator from the
  `\"Geometric Deep Learning on Graphs and Manifolds using Mixture
  Model CNNs\" paper."
  {:db/ident        :pyg/GMMConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1611.08402"})

(def GatedGraphConv
  "The gated graph convolution operator from the \"Gated Graph
  Sequence Neural Networks\" paper."
  {:db/ident        :pyg/GatedGraphConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1511.05493"})

(def GeneralConv
  "A general GNN layer adapted from the `\"Design Space for Graph
  Neural Networks\" paper."
  {:db/ident        :pyg/GeneralConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2011.08843"})

(def GraphConv
  "The graph neural network operator from the `\"Weisfeiler and Leman
  Go Neural: Higher-order Graph Neural Networks\" paper."
  {:db/ident        :pyg/GraphConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1810.02244"})

(def GravNetConv
  "The GravNet operator from the `\"Learning Representations of
  Irregular Particle-detector Geometry with Distance-weighted Graph
  Networks\" paper, where the graph is dynamically constructed using
  nearest neighbors. The neighbors are constructed in a learnable
  low-dimensional projection of the feature space. A second projection
  of the input feature space is then propagated from the neighbors to
  each vertex using distance weights that are derived by applying a
  Gaussian function to the distances."
  {:db/ident        :pyg/GravNetConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1902.07987"})

(def HANConv
  "The Heterogenous Graph Attention Operator from the \"Heterogenous
  Graph Attention Network\" paper."
  {:db/ident        :pyg/HANConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/pdf/1903.07293.pdf"})

(def HEATConv
  "The heterogeneous edge-enhanced graph attentional operator from the
  \"Heterogeneous Edge-Enhanced Graph Attention Network For
  Multi-Agent Trajectory Prediction\" paper, which enhances
  class:`:pyg/GATConv`."
  {:db/ident        :pyg/HEATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2106.07161"})

(def HGTConv
  "The Heterogeneous Graph Transformer (HGT) operator from the
  \"Heterogeneous Graph Transformer\" paper."
  {:db/ident        :pyg/HGTConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2003.01332"})

(def HeteroConv
  "A generic wrapper for computing graph convolution on heterogeneous
  graphs.  This layer will pass messages from source nodes to target
  nodes based on the bipartite GNN layer given for a specific edge
  type."
  {:db/ident        :pyg/HeteroConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def HypergraphConv
  "The hypergraph convolutional operator from the `\"Hypergraph
  Convolution and Hypergraph Attention\" paper."
  {:db/ident        :pyg/HypergraphConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1901.08150"})

(def LEConv
  "The local extremum graph neural network operator from the \"ASAP:
  Adaptive Structure Aware Pooling for Learning Hierarchical Graph
  Representations\" paper, which finds the importance of nodes with
  respect to their neighbors using the difference operator."
  {:db/ident        :pyg/LEConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1911.07979"})

(def LGConv
  "The Light Graph Convolution (LGC) operator from the `\"LightGCN:
  Simplifying and Powering Graph Convolution Network for
  Recommendation\" paper."
  {:db/ident        :pyg/LGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2002.02126"})

(def MFConv
  "The graph neural network operator from the \"Convolutional Networks
  on Graphs for Learning Molecular Fingerprints\" paper."
  {:db/ident        :pyg/MFConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1509.09292"})

(def NNConv
  "The continuous kernel-based convolutional operator from the
  \"Neural Message Passing for Quantum Chemistry\" paper.  This
  convolution is also known as the edge-conditioned convolution from
  the \"Dynamic Edge-Conditioned Filters in Convolutional Neural
  Networks on Graphs\" paper (see class:`:pyg/ECConv` for an alias)."
  {:db/ident        :pyg/NNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    ["https://arxiv.org/abs/1704.01212"
                     "https://arxiv.org/abs/1704.02901"]})

(def PANConv
  "The path integral based convolutional operator from the \"Path
  Integral Based Convolution and Pooling for Graph Neural Networks\"
  paper."
  {:db/ident        :pyg/PANConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2006.16811"})

(def PDNConv
  "The pathfinder discovery network convolutional operator from the
  \"Pathfinder Discovery Networks for Neural Message Passing\" paper."
  {:db/ident        :pyg/PDNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/pdf/2010.12878.pdf"})

(def PNAConv
  "The Principal Neighbourhood Aggregation graph convolution operator
  from the `\"Principal Neighbourhood Aggregation for Graph Nets\"
  paper."
  {:db/ident        :pyg/PNAConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2004.05718"})

(def PPFConv
  "The PPFNet operator from the `\"PPFNet: Global Context Aware Local
  Features for Robust 3D Point Matching\" paper."
  {:db/ident        :pyg/PPFConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1802.02669"})

(def PointConv
  "The PointNet set layer from the `\"PointNet: Deep Learning on Point
  Sets for 3D Classification and Segmentation\" and `\"PointNet++:
  Deep Hierarchical Feature Learning on Point Sets in a Metric Space\"
  papers."
  {:db/ident        :pyg/PointConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    ["https://arxiv.org/abs/1612.00593"
                     "https://arxiv.org/abs/1706.02413"]})

(def PointNetConv
  "The PointNet set layer from the `\"PointNet: Deep Learning on Point
  Sets for 3D Classification and Segmentation\" and `\"PointNet++:
  Deep Hierarchical Feature Learning on Point Sets in a Metric Space\"
  papers."
  {:db/ident        :pyg/PointNetConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    ["https://arxiv.org/abs/1612.00593"
                     "https://arxiv.org/abs/1706.02413"]})

(def PointTransformerConv
  "The Point Transformer layer from the `\"Point Transformer\" paper."
  {:db/ident        :pyg/PointTransformerConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2012.09164"})

(def RGATConv
  "The relational graph attentional operator from the `\"Relational
  Graph Attention Networks\" paper."
  {:db/ident        :pyg/RGATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1904.05811"})

(def RGCNConv
  "The relational graph convolutional operator from the `\"Modeling
  Relational Data with Graph Convolutional Networks\" paper."
  {:db/ident        :pyg/RGCNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1703.06103"})

(def ResGatedGraphConv
  "The residual gated graph convolutional operator from the \"Residual
  Gated Graph ConvNets\" paper."
  {:db/ident        :pyg/ResGatedGraphConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1711.07553"})

(def SAGEConv
  "The GraphSAGE operator from the `\"Inductive Representation
  Learning on Large Graphs\" paper."
  {:db/ident        :pyg/SAGEConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1706.02216"})

(def SGConv
  "The simple graph convolutional operator from the `\"Simplifying
  Graph Convolutional Networks\" paper."
  {:db/ident        :pyg/SGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1902.07153"})

(def SSGConv
  "The simple spectral graph convolutional operator from the \"Simple
  Spectral Graph Convolution\" paper."
  {:db/ident        :pyg/SSGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://openreview.net/forum?id=CYO5T-YjWZV"})

(def SignedConv
  "The signed graph convolutional operator from the `\"Signed Graph
  Convolutional Network\" paper."
  {:db/ident        :pyg/SignedConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1808.06354"})

(def SplineConv
  "The spline-based convolutional operator from the `\"SplineCNN: Fast
  Geometric Deep Learning with Continuous B-Spline Kernels\" paper."
  {:db/ident        :pyg/SplineConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1711.08920"})

(def SuperGATConv
  "The self-supervised graph attentional operator from the `\"How to
  Find Your Friendly Neighborhood: Graph Attention Design with
  Self-Supervision\" paper."
  {:db/ident        :pyg/SuperGATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://openreview.net/forum?id=Wi5KUNlqWty"})

(def TAGConv
  "The topology adaptive graph convolutional networks operator from
  the \"Topology Adaptive Graph Convolutional Networks\" paper."
  {:db/ident        :pyg/TAGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1710.10370"})

(def TransformerConv
  "The graph transformer operator from the `\"Masked Label Prediction:
  Unified Message Passing Model for Semi-Supervised Classification\"
  paper."
  {:db/ident        :pyg/TransformerConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/2009.03509"})

(def WLConv
  "The Weisfeiler Lehman operator from the `\"A Reduction of a Graph
  to a Canonical Form and an Algebra Arising During this Reduction\"
  paper, which iteratively refines node colorings."
  {:db/ident        :pyg/WLConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf"})

(def WLConvContinuous
  "The Weisfeiler Lehman operator from the `\"Wasserstein
  Weisfeiler-Lehman Graph Kernels\" <> paper. Refinement is done
  though a degree-scaled mean aggregation and works on nodes with
  continuous attributes."
  {:db/ident        :pyg/WLConvContinuous,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://arxiv.org/abs/1906.01277"})

(def XConv
  "The convolutional operator on transformed points from the
  `\"PointCNN: Convolution On X-Transformed Points\" paper."
  {:db/ident        :pyg/XConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://arxiv.org/abs/1801.07791"})

(def CuGraphSAGEConv
  "The GraphSAGE operator from the ???Inductive Representation Learning on Large Graphs??? paper."
  {:db/ident        :pyg/CuGraphSAGEConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/CuGraphModule
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.CuGraphSAGEConv.html#torch_geometric.nn.conv.CuGraphSAGEConv"
                     "https://arxiv.org/abs/1706.02216"]})

(def CuGraphGATConv
  {:db/ident        :pyg/CuGraphGATConv
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/CuGraphModule})

;; Aggregation Operators

(def Aggregation
  "An abstract base class for implementing custom aggregations."
  {:db/ident        :pyg/Aggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.Aggregation.html#torch_geometric.nn.aggr.Aggregation"})

(def AttentionalAggregation
  "The soft attention aggregation layer from the `\"Graph Matching
  Networks for Learning the Similarity of Graph Structured Objects\"
  paper."
  {:db/ident        :pyg/AttentionalAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://arxiv.org/abs/1904.12787"})

(def DegreeScalerAggregation
  "Combines one or more aggregators and transforms its output with one
  or more scalers as introduced in the `\"Principal Neighbourhood
  Aggregation for Graph Nets\" paper.  The scalers are normalised by
  the in-degree of the training set and so must be provided at time of
  construction.  See class:`:pyg/PNAConv` for more information."
  {:db/ident        :pyg/DegreeScalerAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://arxiv.org/abs/2004.05718"})

(def EquilibriumAggregation
  "The equilibrium aggregation layer from the `\"Equilibrium
  Aggregation: Encoding Sets via Optimization\" paper."
  {:db/ident        :pyg/EquilibriumAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://arxiv.org/abs/2202.12795"})

(def GraphMultisetTransformer
  "The Graph Multiset Transformer pooling operator from the \"Accurate
  Learning of Graph Representations with Graph Multiset Pooling\"
  paper."
  {:db/ident        :pyg/GraphMultisetTransformer,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://arxiv.org/abs/2102.11533"})

(def LSTMAggregation
  "Performs LSTM-style aggregation in which the elements to aggregate
  are interpreted as a sequence, as described in the `\"Inductive
  Representation Learning on Large Graphs\" paper."
  {:db/ident        :pyg/LSTMAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://arxiv.org/abs/1706.02216"})

(def MaxAggregation
  "An aggregation operator that takes the feature-wise maximum across
  a set of elements."
  {:db/ident        :pyg/MaxAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MeanAggregation
  "An aggregation operator that averages features across a set of
  elements."
  {:db/ident        :pyg/MeanAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MedianAggregation
  "An aggregation operator that returns the feature-wise median of a
  set."
  {:db/ident        :pyg/MedianAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/QuantileAggregation})

(def MinAggregation
  "An aggregation operator that takes the feature-wise minimum across
  a set of elements."
  {:db/ident        :pyg/MinAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MulAggregation
  "An aggregation operator that multiples features across a set of
  elements."
  {:db/ident        :pyg/MulAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MultiAggregation
  "Performs aggregations with one or more aggregators and combines
  aggregated results, as described in the `\"Principal Neighbourhood
  Aggregation for Graph Nets\" and \"Adaptive Filters and Aggregator
  Fusion for Efficient Graph Convolutions\" papers."
  {:db/ident        :pyg/MultiAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    ["https://arxiv.org/abs/2004.05718"
                     "https://arxiv.org/abs/2104.01481"]})

(def PowerMeanAggregation
  "The powermean aggregation operator based on a power term, as
  described in the `\"DeeperGCN: All You Need to Train Deeper GCNs\"
  paper."
  {:db/ident        :pyg/PowerMeanAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://arxiv.org/abs/2006.07739"})

(def QuantileAggregation
  "An aggregation operator that returns the feature-wise `q`-th
  quantile of a set X."
  {:db/ident        :pyg/QuantileAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def Set2Set
  "The Set2Set aggregation operator based on iterative content-based
  attention, as described in the `\"Order Matters: Sequence to
  sequence for Sets\" paper."
  {:db/ident        :pyg/Set2Set,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://arxiv.org/abs/1511.06391"})

(def SoftmaxAggregation
  "The softmax aggregation operator based on a temperature term, as
  described in the `\"DeeperGCN: All You Need to Train Deeper GCNs\"
  paper."
  {:db/ident        :pyg/SoftmaxAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://arxiv.org/abs/2006.07739"})

(def SortAggregation
  "The pooling operator from the `\"An End-to-End Deep Learning
  Architecture for Graph Classification\" paper, where node features
  are sorted in descending order based on their last feature
  channel. The first `k` nodes form the output of the layer."
  {:db/ident        :pyg/SortAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation
   :rdfs/seeAlso    "https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf"})

(def StdAggregation
  "An aggregation operator that takes the feature-wise standard
  deviation across a set of elements."
  {:db/ident        :pyg/StdAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def SumAggregation
  "An aggregation operator that sums up features across a set of
  elements."
  {:db/ident        :pyg/SumAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def VarAggregation
  "An aggregation operator that takes the feature-wise variance across
  a set of elements."
  {:db/ident        :pyg/VarAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

;; Normalization Layers

(def BatchNorm
  "Applies batch normalization over a batch of node features as
  described in the `\"Batch Normalization: Accelerating Deep Network
  Training by Reducing Internal Covariate Shift\" paper."
  {:db/ident        :pyg/BatchNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1502.03167"]})

(def DiffGroupNorm
  "The differentiable group normalization layer from the `\"Towards
  Deeper Graph Neural Networks with Differentiable Group
  Normalization\" paper, which normalizes node features group-wise via
  a learnable soft cluster assignment."
  {:db/ident        :pyg/DiffGroupNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2006.06972"]})

(def GraphNorm
  "Applies graph normalization over individual graphs as described in
  the \"GraphNorm: A Principled Approach to Accelerating Graph Neural
  Network Training\" paper."
  {:db/ident        :pyg/GraphNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2009.03294"]})

(def GraphSizeNorm
  "Applies Graph Size Normalization over each individual graph in a
  batch of node features as described in the \"Benchmarking Graph
  Neural Networks\" paper."
  {:db/ident        :pyg/GraphSizeNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2003.00982"]})

(def InstanceNorm
  "Applies instance normalization over each individual example in a
  batch of node features as described in the `\"Instance
  Normalization: The Missing Ingredient for Fast Stylization\" paper."
  {:db/ident        :pyg/InstanceNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1607.08022"]})

(def LayerNorm
  "Applies layer normalization over each individual example in a batch
  of node features as described in the `\"Layer Normalization\"
  paper."
  {:db/ident        :pyg/LayerNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1607.06450"]})

(def MeanSubtractionNorm
  "Applies layer normalization by subtracting the mean from the inputs
  as described in the `\"Revisiting 'Over-smoothing' in Deep GCNs\"
  paper."
  {:db/ident        :pyg/MeanSubtractionNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/pdf/2003.13663.pdf"]})

(def MessageNorm
  "Applies message normalization over the aggregated messages as
  described in the `\"DeeperGCNs: All You Need to Train Deeper GCNs\"
  paper."
  {:db/ident        :pyg/MessageNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2006.07739"]})

(def PairNorm
  "Applies pair normalization over node features as described in the
  \"PairNorm: Tackling Oversmoothing in GNNs\" paper."
  {:db/ident        :pyg/PairNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1909.12223"]})

;; Pooling Layers

(def ASAPooling
  "The Adaptive Structure Aware Pooling operator from the `\"ASAP:
  Adaptive Structure Aware Pooling for Learning Hierarchical Graph
  Representations\" paper."
  {:db/ident        :pyg/ASAPooling,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1911.07979"]})

(def EdgePooling
  "The edge pooling operator from the `\"Towards Graph Pooling by Edge
  Contraction\" and `\"Edge Contraction Pooling for Graph Neural
  Networks\" papers."
  {:db/ident        :pyg/EdgePooling,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://graphreason.github.io/papers/17.pdf"
                     "https://arxiv.org/abs/1905.10990"]})

(def MemPooling
  "Memory based pooling layer from `\"Memory-Based Graph Networks\"
  paper, which learns a coarsened graph representation based on soft
  cluster assignments."
  {:db/ident        :pyg/MemPooling,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2002.09518"]})


(def PANPooling
  "The path integral based pooling operator from the `\"Path Integral
  Based Convolution and Pooling for Graph Neural Networks\" paper.
  PAN pooling performs top `k` pooling where global node importance is
  measured based on node features and the MET matrix."
  {:db/ident        :pyg/PANPooling,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2006.16811"]})

(def SAGPooling
  "The self-attention pooling operator from the `\"Self-Attention
  Graph Pooling\" and `\"Understanding Attention and Generalization in
  Graph Neural Networks\" papers."
  {:db/ident        :pyg/SAGPooling,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1904.08082"
                     "https://arxiv.org/abs/1905.02850"]})


(def TopKPooling
  "TopK pooling operator from the `\"Graph U-Nets\", `\"Towards Sparse
  Hierarchical Graph Classifiers\" and `\"Understanding Attention and
  Generalization in Graph Neural Networks\" papers."
  {:db/ident        :pyg/TopKPooling,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1905.05178"
                     "https://arxiv.org/abs/1811.01287"
                     "https://arxiv.org/abs/1905.02850"],})

;; Models

(def BasicGNN
  "An abstract class for implementing basic GNN models."
  {:db/ident        :pyg/BasicGNN
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Module})

(def ARGA
  "The Adversarially Regularized Graph Auto-Encoder model from the
  `Adversarially Regularized Graph Autoencoder for Graph Embedding`
  paper."
  {:db/ident        :pyg/ARGA,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/GAE
   :rdfs/seeAlso    "https://arxiv.org/abs/1802.04407"})

(def ARGVA
  "The Adversarially Regularized Variational Graph Auto-Encoder model
  from the `Adversarially Regularized Graph Autoencoder for Graph
  Embedding` paper."
  {:db/ident        :pyg/ARGVA,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/ARGA
   :rdfs/seeAlso    "https://arxiv.org/abs/1802.04407"})

(def AttentiveFP
  "The Attentive FP model for molecular representation learning from
  the `Pushing the Boundaries of Molecular Representation for Drug
  Discovery with the Graph Attention Mechanism` paper, based on graph
  attention mechanisms."
  {:db/ident        :pyg/AttentiveFP,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959"})

(def CorrectAndSmooth
  "The correct and smooth (C&S) post-processing model from the
  `\"Combining Label Propagation And Simple Models Out-performs Graph
  Neural Networks\" paper."
  {:db/ident        :pyg/CorrectAndSmooth,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2010.13993"]})

(def DeepGCNLayer
  "The skip connection operations from the `\"DeepGCNs: Can GCNs Go as
  Deep as CNNs?\" and `\"All You Need to Train Deeper GCNs\"
  papers. The implemented skip connections includes the pre-activation
  residual connection (`\"res+\"`), the residual connection
  (`\"res\"`), the dense connection (`\"dense\"`) and no connections
  (`\"plain\"`)."
  {:db/ident        :pyg/DeepGCNLayer,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1904.03751"
                     "https://arxiv.org/abs/2006.07739"]})

(def DeepGraphInfomax
  "The Deep Graph Infomax model from the \"Deep Graph Infomax\" paper."
  {:db/ident        :pyg/DeepGraphInfomax,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1809.10341"]})

(def DimeNet
  "The directional message passing neural network (DimeNet) from the
  \"Directional Message Passing for Molecular Graphs\" paper. DimeNet
  transforms messages based on the angle between them in a
  rotation-equivariant fashion."
  {:db/ident        :pyg/DimeNet,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2003.03123"]})

(def DimeNetPlusPlus
  "The DimeNet++ from the `\"Fast and Uncertainty-Aware Directional
  Message Passing for Non-Equilibrium Molecules\" paper."
  {:db/ident        :pyg/DimeNetPlusPlus,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DimeNet
   :rdfs/seeAlso    ["https://arxiv.org/abs/2011.14115"]})

(def EdgeCNN
  "The Graph Neural Network from the `\"Dynamic Graph CNN for Learning
  on Point Clouds\" paper, using the class:`:pyg/EdgeConv` operator
  for message passing."
  {:db/ident        :pyg/EdgeCNN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN
   :rdfs/seeAlso    ["https://arxiv.org/abs/1801.07829"]})

(def Explainer
  "An abstract class for integrating explainability into Graph Neural
  Networks.  It also provides general visualization methods for graph
  attributions."
  {:db/ident        :pyg/Explainer,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def GAE
  "The Graph Auto-Encoder model from the `Variational Graph
  Auto-Encoders paper based on user-defined encoder and decoder
  models."
  {:db/ident        :pyg/GAE,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://arxiv.org/abs/1611.07308"})

(def GAT
  "The Graph Neural Network from `\"Graph Attention Networks\" or
  `\"How Attentive are Graph Attention Networks?\" papers, using the
  class:`:pyg/GATConv` or class:`:pyg/GATv2Conv` operator for message
  passing, respectively."
  {:db/ident        :pyg/GAT,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN
   :rdfs/seeAlso    ["https://arxiv.org/abs/1710.10903"
                     "https://arxiv.org/abs/2105.14491"]})

(def GCN
  "The Graph Neural Network from the `\"Semi-supervised Classification
  with Graph Convolutional Networks\" paper, using the
  class:`:pyg/GCNConv` operator for message passing."
  {:db/ident        :pyg/GCN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN
   :rdfs/seeAlso    ["https://arxiv.org/abs/1609.02907"]})

(def GIN
  "The Graph Neural Network from the `\"How Powerful are Graph Neural
  Networks?\" paper, using the class:`:pyg/GINConv` operator for
  message passing."
  {:db/ident        :pyg/GIN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN
   :rdfs/seeAlso    ["https://arxiv.org/abs/1810.00826"]})

(def GraphSAGE
  "The Graph Neural Network from the `\"Inductive Representation
  Learning on Large Graphs\" paper, using the class:`:pyg/SAGEConv`
  operator for message passing."
  {:db/ident        :pyg/GraphSAGE,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN
   :rdfs/seeAlso    ["https://arxiv.org/abs/1706.02216"]})

(def GraphUNet
  "The Graph U-Net model from the `\"Graph U-Nets\" paper which
  implements a U-Net like architecture with graph pooling and
  unpooling operations."
  {:db/ident        :pyg/GraphUNet,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1905.05178"]})

(def InvertibleModule
  "An abstract base class for implementing invertible modules."
  {:db/ident        :pyg/InvertibleModule
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Module})

(def GroupAddRev
  "The Grouped Reversible GNN module from the `\"Graph Neural Networks
  with 1000 Layers\" paper. This module enables training of arbitary
  deep GNNs with a memory complexity independent of the number of
  layers."
  {:db/ident        :pyg/GroupAddRev,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/InvertibleModule
   :rdfs/seeAlso    ["https://arxiv.org/abs/2106.07476"]})

(def InnerProductDecoder
  "The inner product decoder from the `\"Variational Graph
  Auto-Encoders\" paper."
  {:db/ident        :pyg/InnerProductDecoder,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1611.07308"]})

(def JumpingKnowledge
  "The Jumping Knowledge layer aggregation module from the
  \"Representation Learning on Graphs with Jumping Knowledge
  Networks\" paper."
  {:db/ident        :pyg/JumpingKnowledge,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1806.03536"]})

(def LINKX
  "The LINKX model from the `\"Large Scale Learning on Non-Homophilous
  Graphs: New Benchmarks and Strong Simple Methods\" paper."
  {:db/ident        :pyg/LINKX,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2110.14446"]})

(def LabelPropagation
  "The label propagation operator from the `\"Learning from Labeled
  and Unlabeled Data with Label Propagation\" paper."
  {:db/ident        :pyg/LabelPropagation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    ["http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf"]})

(def LightGCN
  "The LightGCN model from the `\"LightGCN: Simplifying and Powering
  Graph Convolution Network for Recommendation\" paper."
  {:db/ident        :pyg/LightGCN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2002.02126"]})

(def MLP
  "A Multi-Layer Perception (MLP) model."
  {:db/ident        :pyg/MLP,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def MaskLabel
  "The label embedding and masking layer from the `\"Masked Label
  Prediction: Unified Message Passing Model for Semi-Supervised
  Classification\" paper."
  {:db/ident        :pyg/MaskLabel,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2009.03509"]})

(def MetaPath2Vec
  "The MetaPath2Vec model from the `\"metapath2vec: Scalable
  Representation Learning for Heterogeneous Networks\" paper where
  random walks based a given `metapath` are sampled in a heterogeneous
  graph, and node embeddings are learned via negative sampling
  optimization."
  {:db/ident        :pyg/MetaPath2Vec,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf"]})

(def Node2Vec
  "The Node2Vec model from the \"node2vec: Scalable Feature Learning
  for Networks\" paper where random walks of length `walk_length` are
  sampled in a given graph, and node embeddings are learned via
  negative sampling optimization."
  {:db/ident        :pyg/Node2Vec,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1607.00653"]})

(def PNA
  "The Graph Neural Network from the `\"Principal Neighbourhood
  Aggregation for Graph Nets\" paper, using the class:`:pyg/PNAConv`
  operator for message passing."
  {:db/ident        :pyg/PNA,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN
   :rdfs/seeAlso    ["https://arxiv.org/abs/2004.05718"]})

(def RECT_L
  "The RECT model, *i.e.* its supervised RECT-L part, from the
  \"Network Embedding with Completely-imbalanced Labels\" paper.  In
  particular, a GCN model is trained that reconstructs semantic class
  knowledge."
  {:db/ident        :pyg/RECT_L,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2007.03545"]})

(def RENet
  "The Recurrent Event Network model from the `\"Recurrent Event
  Network for Reasoning over Temporal Knowledge Graphs\" paper."
  {:db/ident        :pyg/RENet,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1904.05530"]})

(def SchNet
  "The continuous-filter convolutional neural network SchNet from the
  \"SchNet: A Continuous-filter Convolutional Neural Network for
  Modeling Quantum Interactions\" paper."
  {:db/ident        :pyg/SchNet,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1706.08566"]})

(def SignedGCN
  "The signed graph convolutional network model from the `\"Signed
  Graph Convolutional Network\" <> paper.  Internally, this module
  uses the class:`:pyg/SignedConv` operator."
  {:db/ident        :pyg/SignedGCN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1808.06354"]})

(def TGNMemory
  "The Temporal Graph Network (TGN) memory model from the \"Temporal
  Graph Networks for Deep Learning on Dynamic Graphs\" paper."
  {:db/ident        :pyg/TGNMemory,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/2006.10637"]})

(def VGAE
  "The Variational Graph Auto-Encoder model from the \"Variational
  Graph Auto-Encoders\" paper."
  {:db/ident        :pyg/VGAE,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/GAE
   :rdfs/seeAlso    ["https://arxiv.org/abs/1611.07308"]})

(def APPNP
  "The approximate personalized propagation of neural predictions
  layer from the ???Predict then Propagate: Graph Neural Networks meet
  Personalized PageRank??? paper."
  {:db/ident        :pyg/APPNP,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.APPNP.html#torch_geometric.nn.conv.APPNP"
                     "https://arxiv.org/abs/1810.05997"]})

(def MetaLayer
  "A meta layer for building any kind of graph network, inspired by
  the ???Relational Inductive Biases, Deep Learning, and Graph Networks???
  paper."
  {:db/ident        :pyg/MetaLayer,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/Module
   :rdfs/seeAlso    ["https://arxiv.org/abs/1806.01261"]})

;; KGE Models

(def KGEModel
  "An abstract base class for implementing custom KGE models."
  {:db/ident        :pyg/KGEModel
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.KGEModel.html#torch_geometric.nn.kge.KGEModel"})

(def TransE
  "The TransE model from the ???Translating Embeddings for Modeling
  Multi-Relational Data??? paper."
  {:db/ident        :pyg/KGEModel
   :rdf/type        :owl/Class
   :rdfs/subClassOf :pyg/KGEModel
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.TransE.html#torch_geometric.nn.kge.TransE"
                     "https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf"]})

;; Encodings

(def PositionalEncoding
  "The positional encoding scheme from the ???Attention Is All You Need???
  paper."
  {:db/ident        :pyg/PositionalEncoding,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.encoding.PositionalEncoding"
                     "https://arxiv.org/pdf/1706.03762.pdf"]})

(def TemporalEncoding
  "The time-encoding function from the ???Do We Really Need Complicated
  Model Architectures for Temporal Networks???? paper. TemporalEncoding
  first maps each entry to a vector with monotonically exponentially
  decreasing values, and then uses the cosine function to project all
  values to range."
  {:db/ident        :pyg/TemporalEncoding,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.encoding.TemporalEncoding"
                     "https://openreview.net/forum?id=ayPPc0SyLv1"]})

;; Dense Convolutional Layers

(def DenseGCNConv
  "See `:pyg/GCNConv`."
  {:db/ident        :pyg/DenseGCNConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    :pyg/GCNConv})

(def DenseGINConv
  "See `:pyg/GINConv`."
  {:db/ident        :pyg/DenseGINConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    :pyg/GINConv})

(def DenseGraphConv
  "See `:pyg/GraphConv`."
  {:db/ident        :pyg/DenseGraphConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    :pyg/GraphConv})

(def DenseSAGEConv
  "See `:pyg/SAGEConv`."
  {:db/ident        :pyg/DenseSAGEConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    :pyg/SAGEConv})

;; Dense Pooling Layers

(def DMoNPooling
  "The spectral modularity pooling operator from the ???Graph Clustering
  with Graph Neural Networks??? paper"
  {:db/ident        :pyg/DMoNPooling,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.DMoNPooling.html#torch_geometric.nn.dense.DMoNPooling"
                     "https://arxiv.org/abs/2006.16904"]})

;; Model Transformations

(def Transformer
  "A Transformer executes an FX graph node-by-node, applies
  transformations to each node, and produces a new torch.nn.Module. It
  exposes a transform() method that returns the transformed
  Module. Transformer works entirely symbolically."
  {:db/ident        :pyg/Transformer
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.fx.Transformer"})

;; DataParallel Layers

(def DataParallel
  "Implements data parallelism at the module level.

  This container parallelizes the application of the given module by
  splitting a list of :pyg/Data objects and copying them as :pyg/Batch
  objects to each device. In the forward pass, the module is
  replicated on each device, and each replica handles a portion of the
  input. During the backwards pass, gradients from each replica are
  summed into the original module.

  The batch size should be larger than the number of GPUs used.

  The parallelized module must have its parameters and buffers on
  device_ids[0]."
  {:db/ident        :pyg/DataParallel
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/DataParallel})

;; torch_geometric.transforms

(def BaseTransform
  "An abstract base class for writing transforms."
  {:db/ident        :pyg/BaseTransform,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :py/Object})

(def AddLaplacianEigenvectorPE
  "Adds the Laplacian eigenvector positional encoding from the    `\"Benchmarking Graph Neural Networks\" <https://arxiv.org/abs/2003.00982>`_    paper to the given graph    (functional name: :`add_laplacian_eigenvector_pe`)."
  {:db/ident        :pyg/AddLaplacianEigenvectorPE,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def AddMetaPaths
  "Adds additional edge types to a    :`:pyg/HeteroData` object between the source node    type and the destination node type of a given :`metapath`, as described    in the `\"Heterogenous Graph Attention Networks\"    <https://arxiv.org/abs/1903.07293>`_ paper    (functional name: :`add_metapaths`).    Meta-path based neighbors can exploit different aspects of structure    information in heterogeneous graphs.    Formally, a metapath is a path of the form"
  {:db/ident        :pyg/AddMetaPaths,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def AddRandomMetaPaths
  "Adds additional edge types similar to :`AddMetaPaths`.    The key difference is that the added edge type is given by    multiple random walks along the metapath.    One might want to increase the number of random walks    via :`walks_per_node` to achieve competitive performance with    :`AddMetaPaths`."
  {:db/ident        :pyg/AddRandomMetaPaths,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def AddRandomWalkPE
  "Adds the random walk positional encoding from the `\"Graph Neural    Networks with Learnable Structural and Positional Representations\"    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph    (functional name: :`add_random_walk_pe`)."
  {:db/ident        :pyg/AddRandomWalkPE,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def AddSelfLoops
  "Adds self-loops to the given homogeneous or heterogeneous graph    (functional name: :`add_self_loops`)."
  {:db/ident        :pyg/AddSelfLoops,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def Cartesian
  "Saves the relative Cartesian coordinates of linked nodes in its edge    attributes (functional name: :`cartesian`)."
  {:db/ident        :pyg/Cartesian,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def Center
  "Centers node positions :`pos` around the origin    (functional name: :`center`)."
  {:db/ident        :pyg/Center,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def Compose
  "Composes several transforms together."
  {:db/ident        :pyg/Compose,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def Constant
  "Appends a constant value to each node feature :`x`    (functional name: :`constant`)."
  {:db/ident        :pyg/Constant,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def Delaunay
  "Computes the delaunay triangulation of a set of points    (functional name: :`delaunay`)."
  {:db/ident        :pyg/Delaunay,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def Distance
  "Saves the Euclidean distance of linked nodes in its edge attributes    (functional name: :`distance`)."
  {:db/ident        :pyg/Distance,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def FaceToEdge
  "Converts mesh faces :`[3, num_faces]` to edge indices    :`[2, num_edges]` (functional name: :`face_to_edge`)."
  {:db/ident        :pyg/FaceToEdge,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def FeaturePropagation
  "The feature propagation operator from the `\"On the Unreasonable    Effectiveness of Feature propagation in Learning on Graphs with Missing    Node Features\" <https://arxiv.org/abs/2111.12128>`_ paper    (functional name: :`feature_propagation`)"
  {:db/ident        :pyg/FeaturePropagation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def FixedPoints
  "Samples a fixed number of :`num` points and features from a point    cloud (functional name: :`fixed_points`)."
  {:db/ident        :pyg/FixedPoints,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def GCNNorm
  "Applies the GCN normalization from the `\"Semi-supervised Classification    with Graph Convolutional Networks\" <https://arxiv.org/abs/1609.02907>`_    paper (functional name: :`gcn_norm`)."
  {:db/ident        :pyg/GCNNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def GDC
  "Processes the graph via Graph Diffusion Convolution (GDC) from the    `\"Diffusion Improves Graph Learning\" <https://www.kdd.in.tum.de/gdc>`_    paper (functional name: :`gdc`)."
  {:db/ident        :pyg/GDC,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def GenerateMeshNormals
  "Generate normal vectors for each mesh node based on neighboring    faces (functional name: :`generate_mesh_normals`)."
  {:db/ident        :pyg/GenerateMeshNormals,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def GridSampling
  "Clusters points into voxels with size :attr:`size`    (functional name: :`grid_sampling`).    Each cluster returned is a new point based on the mean of all points    inside the given cluster."
  {:db/ident        :pyg/GridSampling,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def IndexToMask
  "Converts indices to a mask representation    (functional name: :`index_to_mask`)."
  {:db/ident        :pyg/IndexToMask,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def KNNGraph
  "Creates a k-NN graph based on node positions :`pos`    (functional name: :`knn_graph`)."
  {:db/ident        :pyg/KNNGraph,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def LaplacianLambdaMax
  "Computes the highest eigenvalue of the graph Laplacian given by    :meth:`torch_geometric.utils.get_laplacian`    (functional name: :`laplacian_lambda_max`)."
  {:db/ident        :pyg/LaplacianLambdaMax,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def LargestConnectedComponents
  "Selects the subgraph that corresponds to the    largest connected components in the graph    (functional name: :`largest_connected_components`)."
  {:db/ident        :pyg/LargestConnectedComponents,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def LineGraph
  "Converts a graph to its corresponding line-graph    (functional name: :`line_graph`):"
  {:db/ident        :pyg/LineGraph,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def LinearTransformation
  "Transforms node positions :`pos` with a square transformation    matrix computed offline (functional name: :`linear_transformation`)"
  {:db/ident        :pyg/LinearTransformation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def LocalCartesian
  "Saves the relative Cartesian coordinates of linked nodes in its edge    attributes (functional name: :`local_cartesian`). Each coordinate gets    *neighborhood-normalized* to the interval :math:`{[0, 1]}^D`."
  {:db/ident        :pyg/LocalCartesian,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def LocalDegreeProfile
  "Appends the Local Degree Profile (LDP) from the `\"A Simple yet    Effective Baseline for Non-attribute Graph Classification\"    <https://arxiv.org/abs/1811.03508>`_ paper    (functional name: :`local_degree_profile`)"
  {:db/ident        :pyg/LocalDegreeProfile,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def MaskToIndex
  "Converts a mask to an index representation    (functional name: :`mask_to_index`)."
  {:db/ident        :pyg/MaskToIndex,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def NormalizeFeatures
  "Row-normalizes the attributes given in :`attrs` to sum-up to one    (functional name: :`normalize_features`)."
  {:db/ident        :pyg/NormalizeFeatures,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def NormalizeRotation
  "Rotates all points according to the eigenvectors of the point cloud    (functional name: :`normalize_rotation`).    If the data additionally holds normals saved in :`data.normal`, these    will be rotated accordingly."
  {:db/ident        :pyg/NormalizeRotation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def NormalizeScale
  "Centers and normalizes node positions to the interval :math:`(-1, 1)`    (functional name: :`normalize_scale`)."
  {:db/ident        :pyg/NormalizeScale,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def OneHotDegree
  "Adds the node degree as one hot encodings to the node features    (functional name: :`one_hot_degree`)."
  {:db/ident        :pyg/OneHotDegree,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def PointPairFeatures
  "Computes the rotation-invariant Point Pair Features    (functional name: :`point_pair_features`)"
  {:db/ident        :pyg/PointPairFeatures,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def Polar
  "Saves the polar coordinates of linked nodes in its edge attributes    (functional name: :`polar`)."
  {:db/ident        :pyg/Polar,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RadiusGraph
  "Creates edges based on node positions :`pos` to all points within a    given distance (functional name: :`radius_graph`)."
  {:db/ident        :pyg/RadiusGraph,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RandomFlip
  "Flips node positions along a given axis randomly with a given    probability (functional name: :`random_flip`)."
  {:db/ident        :pyg/RandomFlip,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RandomJitter
  "Translates node positions by randomly sampled translation values    within a given interval (functional name: :`random_jitter`).    In contrast to other random transformations,    translation is applied separately at each position"
  {:db/ident        :pyg/RandomJitter,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RandomLinkSplit
  "Performs an edge-level random split into training, validation and test    sets of a :`:pyg/Data` or a    :`:pyg/HeteroData` object    (functional name: :`random_link_split`).    The split is performed such that the training split does not include edges    in validation and test splits; and the validation split does not include    edges in the test split."
  {:db/ident        :pyg/RandomLinkSplit,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RandomNodeSplit
  "Performs a node-level random split by adding :`train_mask`,    :`val_mask` and :`test_mask` attributes to the    :`:pyg/Data` or    :`:pyg/HeteroData` object    (functional name: :`random_node_split`)."
  {:db/ident        :pyg/RandomNodeSplit,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RandomRotate
  "Rotates node positions around a specific axis by a randomly sampled    factor within a given interval (functional name: :`random_rotate`)."
  {:db/ident        :pyg/RandomRotate,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RandomScale
  "Scales node positions by a randomly sampled factor :math:`s` within a    given interval, *e.g.*, resulting in the transformation matrix    (functional name: :`random_scale`)"
  {:db/ident        :pyg/RandomScale,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RandomShear
  "Shears node positions by randomly sampled factors :math:`s` within a    given interval, *e.g.*, resulting in the transformation matrix    (functional name: :`random_shear`)"
  {:db/ident        :pyg/RandomShear,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RemoveIsolatedNodes
  "Removes isolated nodes from the graph    (functional name: :`remove_isolated_nodes`)."
  {:db/ident        :pyg/RemoveIsolatedNodes,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RemoveTrainingClasses
  "Removes classes from the node-level training set as given by    :`data.train_mask`, *e.g.*, in order to get a zero-shot label scenario    (functional name: :`remove_training_classes`)."
  {:db/ident        :pyg/RemoveTrainingClasses,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RootedSubgraph
  "Base class for implementing rooted subgraph transformations."
  {:db/ident        :pyg/RootedSubgraph,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def RootedSubgraphData
  "A data object describing a homogeneous graph together with each node's\n    rooted subgraph. It contains several additional properties that hold the\n    information to map to batch of every node's rooted subgraph:"
  {:db/ident        :pyg/RootedSubgraphData,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Data})

(def RootedEgoNets
  "Collects rooted :math:`k`-hop EgoNets for each node in the graph, as    described in the `\"From Stars to Subgraphs: Uplifting Any GNN with Local    Structure Awareness\" <https://arxiv.org/abs/2110.03753>`_ paper."
  {:db/ident        :pyg/RootedEgoNets,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/RootedSubgraph})

(def RootedRWSubgraph
  "Collects rooted random-walk based subgraphs for each node in the graph,    as described in the `\"From Stars to Subgraphs: Uplifting Any GNN with Local    Structure Awareness\" <https://arxiv.org/abs/2110.03753>`_ paper."
  {:db/ident        :pyg/RootedRWSubgraph,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/RootedSubgraph})

(def SIGN
  "The Scalable Inception Graph Neural Network module (SIGN) from the    `\"SIGN: Scalable Inception Graph Neural Networks\"    <https://arxiv.org/abs/2004.11198>`_ paper (functional name: :`sign`),    which precomputes the fixed representations"
  {:db/ident        :pyg/SIGN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def SVDFeatureReduction
  "Dimensionality reduction of node features via Singular Value    Decomposition (SVD) (functional name: :`svd_feature_reduction`)."
  {:db/ident        :pyg/SVDFeatureReduction,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def SamplePoints
  "Uniformly samples :`num` points on the mesh faces according to    their face area (functional name: :`sample_points`)."
  {:db/ident        :pyg/SamplePoints,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def Spherical
  "Saves the spherical coordinates of linked nodes in its edge attributes    (functional name: :`spherical`)."
  {:db/ident        :pyg/Spherical,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def TargetIndegree
  "Saves the globally normalized degree of target nodes    (functional name: :`target_indegree`)"
  {:db/ident        :pyg/TargetIndegree,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def ToDense
  "Converts a sparse adjacency matrix to a dense adjacency matrix with    shape :`[num_nodes, num_nodes, *]` (functional name: :`to_dense`)."
  {:db/ident        :pyg/ToDense,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def ToDevice
  "Performs tensor device conversion, either for all attributes of the    :`:pyg/Data` object or only the ones given by    :`attrs` (functional name: :`to_device`)."
  {:db/ident        :pyg/ToDevice,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def ToSLIC
  "Converts an image to a superpixel representation using the    :meth:`skimage.segmentation.slic` algorithm, resulting in a    :`:pyg/Data` object holding the centroids of    superpixels in :`pos` and their mean color in :`x`    (functional name: :`to_slic`)."
  {:db/ident        :pyg/ToSLIC,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def ToSparseTensor
  "Converts the :`edge_index` attributes of a homogeneous or    heterogeneous data object into a (transposed):`torch_sparse.SparseTensor` type with key :`adj_t`    (functional name: :`to_sparse_tensor`)."
  {:db/ident        :pyg/ToSparseTensor,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def ToUndirected
  "Converts a homogeneous or heterogeneous graph to an undirected graph    such that :math:`(j,i) \\in \\mathcal{E}` for every edge    :math:`(i,j) \\in \\mathcal{E}` (functional name: :`to_undirected`).    In heterogeneous graphs, will add \"reverse\" connections for *all* existing    edge types."
  {:db/ident        :pyg/ToUndirected,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def TwoHop
  "Adds the two hop edges to the edge indices    (functional name: :`two_hop`)."
  {:db/ident        :pyg/TwoHop,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})

(def VirtualNode
  "Appends a virtual node to the given homogeneous graph that is connected    to all other nodes, as described in the `\"Neural Message Passing for    Quantum Chemistry\" <https://arxiv.org/abs/1704.01212>`_ paper    (functional name: :`virtual_node`).    The virtual node serves as a global scratch space that each node both reads    from and writes to in every step of message passing.    This allows information to travel long distances during the propagation    phase."
  {:db/ident        :pyg/VirtualNode,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BaseTransform})
