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

;; Convolutional Layers

(def SimpleConv
  "A simple message passing operator that performs (non-trainable) propagation"
  {:db/ident        :pyg/SimpleConv
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SimpleConv.html#torch_geometric.nn.conv.SimpleConv"})

(def AGNNConv
  "The graph attentional propagation layer from the\n    `\"Attention-based Graph Neural Network for Semi-Supervised Learning\"\n    <https://arxiv.org/abs/1803.03735>`_ paper"
  {:db/ident        :pyg/AGNNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def APPNP
  "The approximate personalized propagation of neural predictions layer\n    from the `\"Predict then Propagate: Graph Neural Networks meet Personalized\n    PageRank\" <https://arxiv.org/abs/1810.05997>`_ paper"
  {:db/ident        :pyg/APPNP,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def ARMAConv
  "The ARMA graph convolutional operator from the `\"Graph Neural Networks\n    with Convolutional ARMA Filters\" <https://arxiv.org/abs/1901.01343>`_ paper"
  {:db/ident        :pyg/ARMAConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def CGConv
  "The crystal graph convolutional operator from the\n    `\"Crystal Graph Convolutional Neural Networks for an\n    Accurate and Interpretable Prediction of Material Properties\"\n    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_\n    paper"
  {:db/ident        :pyg/CGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def ChebConv
  "The chebyshev spectral graph convolutional operator from the\n    `\"Convolutional Neural Networks on Graphs with Fast Localized Spectral\n    Filtering\" <https://arxiv.org/abs/1606.09375>`_ paper"
  {:db/ident        :pyg/ChebConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def ClusterGCNConv
  "The ClusterGCN graph convolutional operator from the\n    `\"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph\n    Convolutional Networks\" <https://arxiv.org/abs/1905.07953>`_ paper"
  {:db/ident        :pyg/ClusterGCNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def DNAConv
  "The dynamic neighborhood aggregation operator from the `\"Just Jump:\n    Towards Dynamic Neighborhood Aggregation in Graph Neural Networks\"\n    <https://arxiv.org/abs/1904.04849>`_ paper"
  {:db/ident        :pyg/DNAConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def DynamicEdgeConv
  "The dynamic edge convolutional operator from the `\"Dynamic Graph CNN\n    for Learning on Point Clouds\" <https://arxiv.org/abs/1801.07829>`_ paper\n    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is\n    dynamically constructed using nearest neighbors in the feature space."
  {:db/ident        :pyg/DynamicEdgeConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def ECConv
  "The continuous kernel-based convolutional operator from the\n    `\"Neural Message Passing for Quantum Chemistry\"\n    <https://arxiv.org/abs/1704.01212>`_ paper.\n    This convolution is also known as the edge-conditioned convolution from the\n    `\"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on\n    Graphs\" <https://arxiv.org/abs/1704.02901>`_ paper (see\n    :class:`torch_geometric.nn.conv.ECConv` for an alias):"
  {:db/ident        :pyg/ECConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def EGConv
  "The Efficient Graph Convolution from the `\"Adaptive Filters and\n    Aggregator Fusion for Efficient Graph Convolutions\"\n    <https://arxiv.org/abs/2104.01481>`_ paper."
  {:db/ident        :pyg/EGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def EdgeConv
  "The edge convolutional operator from the `\"Dynamic Graph CNN for\n    Learning on Point Clouds\" <https://arxiv.org/abs/1801.07829>`_ paper"
  {:db/ident        :pyg/EdgeConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def FAConv
  "The Frequency Adaptive Graph Convolution operator from the\n    `\"Beyond Low-Frequency Information in Graph Convolutional Networks\"\n    <https://arxiv.org/abs/2101.00797>`_ paper"
  {:db/ident        :pyg/FAConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def FastRGCNConv
  "See :class:`RGCNConv`."
  {:db/ident        :pyg/FastRGCNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf [:pyg/RGCNConv
                     :pyg/MessagePassing
                     :torch/Module
                     :py/Object]})

(def FeaStConv
  "The (translation-invariant) feature-steered convolutional operator from\n    the `\"FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis\"\n    <https://arxiv.org/abs/1706.05206>`_ paper"
  {:db/ident        :pyg/FeaStConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def FiLMConv
  "The FiLM graph convolutional operator from the\n    `\"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation\"\n    <https://arxiv.org/abs/1906.12192>`_ paper"
  {:db/ident        :pyg/FiLMConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def FusedGATConv
  "The fused graph attention operator from the\n    `\"Understanding GNN Computational Graph: A Coordinated Computation, IO, and\n    Memory Perspective\"\n    <https://proceedings.mlsys.org/paper/2022/file/\n    9a1158154dfa42caddbd0694a4e9bdc8-Paper.pdf>`_ paper."
  {:db/ident        :pyg/FusedGATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/GATConv})

(def GATConv
  "The graph attentional operator from the `\"Graph Attention Networks\"\n    <https://arxiv.org/abs/1710.10903>`_ paper"
  {:db/ident        :pyg/GATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GATv2Conv
  "The GATv2 operator from the `\"How Attentive are Graph Attention\n    Networks?\" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the\n    static attention problem of the standard\n    :class:`~torch_geometric.conv.GATConv` layer.\n    Since the linear layers in the standard GAT are applied right after each\n    other, the ranking of attended nodes is unconditioned on the query node.\n    In contrast, in :class:`GATv2`, every node can attend to any other node."
  {:db/ident        :pyg/GATv2Conv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GCN2Conv
  "The graph convolutional operator with initial residual connections and\n    identity mapping (GCNII) from the `\"Simple and Deep Graph Convolutional\n    Networks\" <https://arxiv.org/abs/2007.02133>`_ paper"
  {:db/ident        :pyg/GCN2Conv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GCNConv
  "The graph convolutional operator from the `\"Semi-supervised\n    Classification with Graph Convolutional Networks\"\n    <https://arxiv.org/abs/1609.02907>`_ paper"
  {:db/ident        :pyg/GCNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GENConv
  "The GENeralized Graph Convolution (GENConv) from the `\"DeeperGCN: All\n    You Need to Train Deeper GCNs\" <https://arxiv.org/abs/2006.07739>`_ paper.\n    Supports SoftMax & PowerMean aggregation. The message construction is:"
  {:db/ident        :pyg/GENConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GINConv
  "The graph isomorphism operator from the `\"How Powerful are\n    Graph Neural Networks?\" <https://arxiv.org/abs/1810.00826>`_ paper"
  {:db/ident        :pyg/GINConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GINEConv
  "The modified :class:`GINConv` operator from the `\"Strategies for\n    Pre-training Graph Neural Networks\" <https://arxiv.org/abs/1905.12265>`_\n    paper"
  {:db/ident        :pyg/GINEConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GMMConv
  "The gaussian mixture model convolutional operator from the `\"Geometric\n    Deep Learning on Graphs and Manifolds using Mixture Model CNNs\"\n    <https://arxiv.org/abs/1611.08402>`_ paper"
  {:db/ident        :pyg/GMMConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GatedGraphConv
  "The gated graph convolution operator from the `\"Gated Graph Sequence\n    Neural Networks\" <https://arxiv.org/abs/1511.05493>`_ paper"
  {:db/ident        :pyg/GatedGraphConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GeneralConv
  "A general GNN layer adapted from the `\"Design Space for Graph Neural\n    Networks\" <https://arxiv.org/abs/2011.08843>`_ paper."
  {:db/ident        :pyg/GeneralConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GraphConv
  "The graph neural network operator from the `\"Weisfeiler and Leman Go\n    Neural: Higher-order Graph Neural Networks\"\n    <https://arxiv.org/abs/1810.02244>`_ paper"
  {:db/ident        :pyg/GraphConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def GravNetConv
  "The GravNet operator from the `\"Learning Representations of Irregular\n    Particle-detector Geometry with Distance-weighted Graph\n    Networks\" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is\n    dynamically constructed using nearest neighbors.\n    The neighbors are constructed in a learnable low-dimensional projection of\n    the feature space.\n    A second projection of the input feature space is then propagated from the\n    neighbors to each vertex using distance weights that are derived by\n    applying a Gaussian function to the distances."
  {:db/ident        :pyg/GravNetConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def HANConv
  "\n    The Heterogenous Graph Attention Operator from the\n    `\"Heterogenous Graph Attention Network\"\n    <https://arxiv.org/pdf/1903.07293.pdf>`_ paper."
  {:db/ident        :pyg/HANConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def HEATConv
  "The heterogeneous edge-enhanced graph attentional operator from the\n    `\"Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent\n    Trajectory Prediction\" <https://arxiv.org/abs/2106.07161>`_ paper, which\n    enhances :class:`~torch_geometric.nn.conv.GATConv` by:"
  {:db/ident        :pyg/HEATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def HGTConv
  "The Heterogeneous Graph Transformer (HGT) operator from the\n    `\"Heterogeneous Graph Transformer\" <https://arxiv.org/abs/2003.01332>`_\n    paper."
  {:db/ident        :pyg/HGTConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def HeteroConv
  "A generic wrapper for computing graph convolution on heterogeneous\n    graphs.\n    This layer will pass messages from source nodes to target nodes based on\n    the bipartite GNN layer given for a specific edge type.\n    If multiple relations point to the same destination, their results will be\n    aggregated according to :attr:`aggr`.\n    In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is\n    especially useful if you want to apply different message passing modules\n    for different edge types."
  {:db/ident        :pyg/HeteroConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def HypergraphConv
  "The hypergraph convolutional operator from the `\"Hypergraph Convolution\n    and Hypergraph Attention\" <https://arxiv.org/abs/1901.08150>`_ paper"
  {:db/ident        :pyg/HypergraphConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def LEConv
  "The local extremum graph neural network operator from the\n    `\"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph\n    Representations\" <https://arxiv.org/abs/1911.07979>`_ paper, which finds\n    the importance of nodes with respect to their neighbors using the\n    difference operator:"
  {:db/ident        :pyg/LEConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def LGConv
  "The Light Graph Convolution (LGC) operator from the `\"LightGCN:\n    Simplifying and Powering Graph Convolution Network for Recommendation\"\n    <https://arxiv.org/abs/2002.02126>`_ paper"
  {:db/ident        :pyg/LGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def MFConv
  "The graph neural network operator from the\n    `\"Convolutional Networks on Graphs for Learning Molecular Fingerprints\"\n    <https://arxiv.org/abs/1509.09292>`_ paper"
  {:db/ident        :pyg/MFConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def NNConv
  "The continuous kernel-based convolutional operator from the\n    `\"Neural Message Passing for Quantum Chemistry\"\n    <https://arxiv.org/abs/1704.01212>`_ paper.\n    This convolution is also known as the edge-conditioned convolution from the\n    `\"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on\n    Graphs\" <https://arxiv.org/abs/1704.02901>`_ paper (see\n    :class:`torch_geometric.nn.conv.ECConv` for an alias):"
  {:db/ident        :pyg/NNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def PANConv
  "The path integral based convolutional operator from the\n    `\"Path Integral Based Convolution and Pooling for Graph Neural Networks\"\n    <https://arxiv.org/abs/2006.16811>`_ paper"
  {:db/ident        :pyg/PANConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def PDNConv
  "The pathfinder discovery network convolutional operator from the\n    `\"Pathfinder Discovery Networks for Neural Message Passing\"\n    <https://arxiv.org/pdf/2010.12878.pdf>`_ paper"
  {:db/ident        :pyg/PDNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def PNAConv
  "The Principal Neighbourhood Aggregation graph convolution operator\n    from the `\"Principal Neighbourhood Aggregation for Graph Nets\"\n    <https://arxiv.org/abs/2004.05718>`_ paper"
  {:db/ident        :pyg/PNAConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def PPFConv
  "The PPFNet operator from the `\"PPFNet: Global Context Aware Local\n    Features for Robust 3D Point Matching\" <https://arxiv.org/abs/1802.02669>`_\n    paper"
  {:db/ident        :pyg/PPFConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def PointConv
  "The PointNet set layer from the `\"PointNet: Deep Learning on Point Sets\n    for 3D Classification and Segmentation\"\n    <https://arxiv.org/abs/1612.00593>`_ and `\"PointNet++: Deep Hierarchical\n    Feature Learning on Point Sets in a Metric Space\"\n    <https://arxiv.org/abs/1706.02413>`_ papers"
  {:db/ident        :pyg/PointConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def PointNetConv
  "The PointNet set layer from the `\"PointNet: Deep Learning on Point Sets\n    for 3D Classification and Segmentation\"\n    <https://arxiv.org/abs/1612.00593>`_ and `\"PointNet++: Deep Hierarchical\n    Feature Learning on Point Sets in a Metric Space\"\n    <https://arxiv.org/abs/1706.02413>`_ papers"
  {:db/ident        :pyg/PointNetConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def PointTransformerConv
  "The Point Transformer layer from the `\"Point Transformer\"\n    <https://arxiv.org/abs/2012.09164>`_ paper"
  {:db/ident        :pyg/PointTransformerConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def RGATConv
  "The relational graph attentional operator from the `\"Relational Graph\n    Attention Networks\" <https://arxiv.org/abs/1904.05811>`_ paper.\n    Here, attention logits :math:`\\mathbf{a}^{(r)}_{i,j}` are computed for each\n    relation type :math:`r` with the help of both query and key kernels, *i.e.*"
  {:db/ident        :pyg/RGATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def RGCNConv
  "The relational graph convolutional operator from the `\"Modeling\n    Relational Data with Graph Convolutional Networks\"\n    <https://arxiv.org/abs/1703.06103>`_ paper"
  {:db/ident        :pyg/RGCNConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def ResGatedGraphConv
  "The residual gated graph convolutional operator from the\n    `\"Residual Gated Graph ConvNets\" <https://arxiv.org/abs/1711.07553>`_ paper"
  {:db/ident        :pyg/ResGatedGraphConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def SAGEConv
  "The GraphSAGE operator from the `\"Inductive Representation Learning on\n    Large Graphs\" <https://arxiv.org/abs/1706.02216>`_ paper"
  {:db/ident        :pyg/SAGEConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def SGConv
  "The simple graph convolutional operator from the `\"Simplifying Graph\n    Convolutional Networks\" <https://arxiv.org/abs/1902.07153>`_ paper"
  {:db/ident        :pyg/SGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def SSGConv
  "The simple spectral graph convolutional operator from the\n    `\"Simple Spectral Graph Convolution\"\n    <https://openreview.net/forum?id=CYO5T-YjWZV>`_ paper"
  {:db/ident        :pyg/SSGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def SignedConv
  "The signed graph convolutional operator from the `\"Signed Graph\n    Convolutional Network\" <https://arxiv.org/abs/1808.06354>`_ paper"
  {:db/ident        :pyg/SignedConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def SplineConv
  "The spline-based convolutional operator from the `\"SplineCNN: Fast\n    Geometric Deep Learning with Continuous B-Spline Kernels\"\n    <https://arxiv.org/abs/1711.08920>`_ paper"
  {:db/ident        :pyg/SplineConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def SuperGATConv
  "The self-supervised graph attentional operator from the `\"How to Find\n    Your Friendly Neighborhood: Graph Attention Design with Self-Supervision\"\n    <https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper"
  {:db/ident        :pyg/SuperGATConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def TAGConv
  "The topology adaptive graph convolutional networks operator from the\n    `\"Topology Adaptive Graph Convolutional Networks\"\n    <https://arxiv.org/abs/1710.10370>`_ paper"
  {:db/ident        :pyg/TAGConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def TransformerConv
  "The graph transformer operator from the `\"Masked Label Prediction:\n    Unified Message Passing Model for Semi-Supervised Classification\"\n    <https://arxiv.org/abs/2009.03509>`_ paper"
  {:db/ident        :pyg/TransformerConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def WLConv
  "The Weisfeiler Lehman operator from the `\"A Reduction of a Graph to a\n    Canonical Form and an Algebra Arising During this Reduction\"\n    <https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf>`_ paper, which\n    iteratively refines node colorings:"
  {:db/ident        :pyg/WLConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def WLConvContinuous
  "The Weisfeiler Lehman operator from the `\"Wasserstein\n    Weisfeiler-Lehman Graph Kernels\" <https://arxiv.org/abs/1906.01277>`_\n    paper. Refinement is done though a degree-scaled mean aggregation and\n    works on nodes with continuous attributes:"
  {:db/ident        :pyg/WLConvContinuous,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def XConv
  "The convolutional operator on :math:`\\mathcal{X}`-transformed points\n    from the `\"PointCNN: Convolution On X-Transformed Points\"\n    <https://arxiv.org/abs/1801.07791>`_ paper"
  {:db/ident        :pyg/XConv,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def CuGraphSAGEConv
  "The GraphSAGE operator from the “Inductive Representation Learning on Large Graphs” paper."
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
  "The soft attention aggregation layer from the `\"Graph Matching Networks\n    for Learning the Similarity of Graph Structured Objects\"\n    <https://arxiv.org/abs/1904.12787>`_ paper"
  {:db/ident        :pyg/AttentionalAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def DegreeScalerAggregation
  "Combines one or more aggregators and transforms its output with one or\n    more scalers as introduced in the `\"Principal Neighbourhood Aggregation for\n    Graph Nets\" <https://arxiv.org/abs/2004.05718>`_ paper.\n    The scalers are normalised by the in-degree of the training set and so must\n    be provided at time of construction.\n    See :class:`torch_geometric.nn.conv.PNAConv` for more information."
  {:db/ident        :pyg/DegreeScalerAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def EquilibriumAggregation
  "The equilibrium aggregation layer from the `\"Equilibrium Aggregation:\n    Encoding Sets via Optimization\" <https://arxiv.org/abs/2202.12795>`_ paper.\n    The output of this layer :math:`\\mathbf{y}` is defined implicitly via a\n    potential function :math:`F(\\mathbf{x}, \\mathbf{y})`, a regularization term\n    :math:`R(\\mathbf{y})`, and the condition"
  {:db/ident        :pyg/EquilibriumAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def GraphMultisetTransformer
  "The Graph Multiset Transformer pooling operator from the\n    `\"Accurate Learning of Graph Representations\n    with Graph Multiset Pooling\" <https://arxiv.org/abs/2102.11533>`_ paper."
  {:db/ident        :pyg/GraphMultisetTransformer,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def LSTMAggregation
  "Performs LSTM-style aggregation in which the elements to aggregate are\n    interpreted as a sequence, as described in the `\"Inductive Representation\n    Learning on Large Graphs\" <https://arxiv.org/abs/1706.02216>`_ paper."
  {:db/ident        :pyg/LSTMAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MaxAggregation
  "An aggregation operator that takes the feature-wise maximum across a\n    set of elements"
  {:db/ident        :pyg/MaxAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MeanAggregation
  "An aggregation operator that averages features across a set of elements"
  {:db/ident        :pyg/MeanAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MedianAggregation
  "An aggregation operator that returns the feature-wise median of a set.\n    That is, for every feature :math:`d`, it computes"
  {:db/ident        :pyg/MedianAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/QuantileAggregation})

(def MinAggregation
  "An aggregation operator that takes the feature-wise minimum across a\n    set of elements"
  {:db/ident        :pyg/MinAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MulAggregation
  "An aggregation operator that multiples features across a set of\n    elements"
  {:db/ident        :pyg/MulAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def MultiAggregation
  "Performs aggregations with one or more aggregators and combines\n    aggregated results, as described in the `\"Principal Neighbourhood\n    Aggregation for Graph Nets\" <https://arxiv.org/abs/2004.05718>`_ and\n    `\"Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions\"\n    <https://arxiv.org/abs/2104.01481>`_ papers."
  {:db/ident        :pyg/MultiAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def PowerMeanAggregation
  "The powermean aggregation operator based on a power term, as\n    described in the `\"DeeperGCN: All You Need to Train Deeper GCNs\"\n    <https://arxiv.org/abs/2006.07739>`_ paper"
  {:db/ident        :pyg/PowerMeanAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def QuantileAggregation
  "An aggregation operator that returns the feature-wise :math:`q`-th\n    quantile of a set :math:`\\mathcal{X}`. That is, for every feature\n    :math:`d`, it computes"
  {:db/ident        :pyg/QuantileAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def Set2Set
  "The Set2Set aggregation operator based on iterative content-based\n    attention, as described in the `\"Order Matters: Sequence to sequence for\n    Sets\" <https://arxiv.org/abs/1511.06391>`_ paper"
  {:db/ident        :pyg/Set2Set,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def SoftmaxAggregation
  "The softmax aggregation operator based on a temperature term, as\n    described in the `\"DeeperGCN: All You Need to Train Deeper GCNs\"\n    <https://arxiv.org/abs/2006.07739>`_ paper"
  {:db/ident        :pyg/SoftmaxAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def SortAggregation
  "The pooling operator from the `\"An End-to-End Deep Learning\n    Architecture for Graph Classification\"\n    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,\n    where node features are sorted in descending order based on their last\n    feature channel. The first :math:`k` nodes form the output of the layer."
  {:db/ident        :pyg/SortAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def StdAggregation
  "An aggregation operator that takes the feature-wise standard deviation\n    across a set of elements"
  {:db/ident        :pyg/StdAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def SumAggregation
  "An aggregation operator that sums up features across a set of elements"
  {:db/ident        :pyg/SumAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

(def VarAggregation
  "An aggregation operator that takes the feature-wise variance across a\n    set of elements"
  {:db/ident        :pyg/VarAggregation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/Aggregation})

;; Normalization Layers

(def BatchNorm
  "Applies batch normalization over a batch of node features as described\n    in the `\"Batch Normalization: Accelerating Deep Network Training by\n    Reducing Internal Covariate Shift\" <https://arxiv.org/abs/1502.03167>`_\n    paper"
  {:db/ident        :pyg/BatchNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def DiffGroupNorm
  "The differentiable group normalization layer from the `\"Towards Deeper\n    Graph Neural Networks with Differentiable Group Normalization\"\n    <https://arxiv.org/abs/2006.06972>`_ paper, which normalizes node features\n    group-wise via a learnable soft cluster assignment"
  {:db/ident        :pyg/DiffGroupNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def GraphNorm
  "Applies graph normalization over individual graphs as described in the\n    `\"GraphNorm: A Principled Approach to Accelerating Graph Neural Network\n    Training\" <https://arxiv.org/abs/2009.03294>`_ paper"
  {:db/ident        :pyg/GraphNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def GraphSizeNorm
  "Applies Graph Size Normalization over each individual graph in a batch\n    of node features as described in the\n    `\"Benchmarking Graph Neural Networks\" <https://arxiv.org/abs/2003.00982>`_\n    paper"
  {:db/ident        :pyg/GraphSizeNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def InstanceNorm
  "Applies instance normalization over each individual example in a batch\n    of node features as described in the `\"Instance Normalization: The Missing\n    Ingredient for Fast Stylization\" <https://arxiv.org/abs/1607.08022>`_\n    paper"
  {:db/ident        :pyg/InstanceNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def LayerNorm
  "Applies layer normalization over each individual example in a batch\n    of node features as described in the `\"Layer Normalization\"\n    <https://arxiv.org/abs/1607.06450>`_ paper"
  {:db/ident        :pyg/LayerNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def MeanSubtractionNorm
  "Applies layer normalization by subtracting the mean from the inputs\n    as described in the  `\"Revisiting 'Over-smoothing' in Deep GCNs\"\n    <https://arxiv.org/pdf/2003.13663.pdf>`_ paper"
  {:db/ident        :pyg/MeanSubtractionNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def MessageNorm
  "Applies message normalization over the aggregated messages as described\n    in the `\"DeeperGCNs: All You Need to Train Deeper GCNs\"\n    <https://arxiv.org/abs/2006.07739>`_ paper"
  {:db/ident        :pyg/MessageNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def PairNorm
  "Applies pair normalization over node features as described in the\n    `\"PairNorm: Tackling Oversmoothing in GNNs\"\n    <https://arxiv.org/abs/1909.12223>`_ paper"
  {:db/ident        :pyg/PairNorm,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

;; Pooling Layers

(def ASAPooling
  {:db/ident        :pyg/ASAPooling,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    [""
                     ""]})

(def EdgePooling
  {:db/ident        :pyg/EdgePooling, :rdf/type :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    [""
                     ""]})

(def MemPooling
  {:db/ident        :pyg/MemPooling, :rdf/type :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    [""
                     ""]})

(def PANPooling
  {:db/ident        :pyg/PANPooling, :rdf/type :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    [""
                     ""]})

(def SAGPooling
  {:db/ident        :pyg/SAGPooling, :rdf/type :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    [""
                     ""]})

(def TopKPooling
  {:db/ident        :pyg/TopKPooling,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    [""
                     ""]})

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
  "The Attentive FP model for molecular representation learning from the   `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism` paper, based on graph attention mechanisms."
  {:db/ident        :pyg/AttentiveFP,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959"})

(def CorrectAndSmooth
  "The correct and smooth (C&S) post-processing model from the\n
  `\"Combining Label Propagation And Simple Models Out-performs Graph
  Neural\n Networks\"\n <https://arxiv.org/abs/2010.13993>`_ paper,
  where soft predictions\n :math:`\\mathbf{Z}` (obtained from a simple
  base predictor) are\n first corrected based on ground-truth
  training\n label information :math:`\\mathbf{Y}` and residual
  propagation."
  {:db/ident        :pyg/CorrectAndSmooth,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def DeepGCNLayer
  "The skip connection operations from the\n `\"DeepGCNs: Can GCNs Go
  as Deep as CNNs?\"\n <https://arxiv.org/abs/1904.03751>`_ and `\"All
  You Need to Train Deeper\n GCNs\"
  <https://arxiv.org/abs/2006.07739>`_ papers.\n The implemented skip
  connections includes the pre-activation residual\n connection
  (:obj:`\"res+\"`), the residual connection (:obj:`\"res\"`),\n the
  dense connection (:obj:`\"dense\"`) and no connections
  (:obj:`\"plain\"`)."
  {:db/ident        :pyg/DeepGCNLayer,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def DeepGraphInfomax
  "The Deep Graph Infomax model from the\n    `\"Deep Graph Infomax\" <https://arxiv.org/abs/1809.10341>`_\n    paper based on user-defined encoder and summary model :math:`\\mathcal{E}`\n    and :math:`\\mathcal{R}` respectively, and a corruption function\n    :math:`\\mathcal{C}`."
  {:db/ident        :pyg/DeepGraphInfomax,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def DimeNet
  "The directional message passing neural network (DimeNet) from the\n    `\"Directional Message Passing for Molecular Graphs\"\n    <https://arxiv.org/abs/2003.03123>`_ paper.\n    DimeNet transforms messages based on the angle between them in a\n    rotation-equivariant fashion."
  {:db/ident        :pyg/DimeNet,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def DimeNetPlusPlus
  "The DimeNet++ from the `\"Fast and Uncertainty-Aware\n    Directional Message Passing for Non-Equilibrium Molecules\"\n    <https://arxiv.org/abs/2011.14115>`_ paper."
  {:db/ident        :pyg/DimeNetPlusPlus,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/DimeNet})

(def EdgeCNN
  "The Graph Neural Network from the `\"Dynamic Graph CNN for Learning on\n    Point Clouds\" <https://arxiv.org/abs/1801.07829>`_ paper, using the\n    :class:`~torch_geometric.nn.conv.EdgeConv` operator for message passing."
  {:db/ident        :pyg/EdgeCNN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN})

(def Explainer
  "An abstract class for integrating explainability into Graph Neural\n    Networks.\n    It also provides general visualization methods for graph attributions."
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
  "The Graph Neural Network from `\"Graph Attention Networks\"\n    <https://arxiv.org/abs/1710.10903>`_ or `\"How Attentive are Graph Attention\n    Networks?\" <https://arxiv.org/abs/2105.14491>`_ papers, using the\n    :class:`~torch_geometric.nn.GATConv` or\n    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,\n    respectively."
  {:db/ident        :pyg/GAT,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN})

(def GCN
  "The Graph Neural Network from the `\"Semi-supervised\n    Classification with Graph Convolutional Networks\"\n    <https://arxiv.org/abs/1609.02907>`_ paper, using the\n    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing."
  {:db/ident        :pyg/GCN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN})

(def GIN
  "The Graph Neural Network from the `\"How Powerful are Graph Neural\n    Networks?\" <https://arxiv.org/abs/1810.00826>`_ paper, using the\n    :class:`~torch_geometric.nn.GINConv` operator for message passing."
  {:db/ident        :pyg/GIN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN})

(def GraphSAGE
  "The Graph Neural Network from the `\"Inductive Representation Learning\n    on Large Graphs\" <https://arxiv.org/abs/1706.02216>`_ paper, using the\n    :class:`~torch_geometric.nn.SAGEConv` operator for message passing."
  {:db/ident        :pyg/GraphSAGE,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN})

(def GraphUNet
  "The Graph U-Net model from the `\"Graph U-Nets\"\n    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like\n    architecture with graph pooling and unpooling operations."
  {:db/ident        :pyg/GraphUNet,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def InvertibleModule
  "An abstract base class for implementing invertible modules."
  {:db/ident        :pyg/InvertibleModule
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Module})

(def GroupAddRev
  "The Grouped Reversible GNN module from the `\"Graph Neural Networks with\n    1000 Layers\" <https://arxiv.org/abs/2106.07476>`_ paper.\n    This module enables training of arbitary deep GNNs with a memory complexity\n    independent of the number of layers."
  {:db/ident        :pyg/GroupAddRev,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/InvertibleModule})

(def InnerProductDecoder
  "The inner product decoder from the `\"Variational Graph Auto-Encoders\"\n    <https://arxiv.org/abs/1611.07308>`_ paper"
  {:db/ident        :pyg/InnerProductDecoder,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def JumpingKnowledge
  "The Jumping Knowledge layer aggregation module from the\n    `\"Representation Learning on Graphs with Jumping Knowledge Networks\"\n    <https://arxiv.org/abs/1806.03536>`_ paper based on either\n    **concatenation** (:obj:`\"cat\"`)"
  {:db/ident        :pyg/JumpingKnowledge,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def LINKX
  "The LINKX model from the `\"Large Scale Learning on Non-Homophilous\n    Graphs: New Benchmarks and Strong Simple Methods\"\n    <https://arxiv.org/abs/2110.14446>`_ paper"
  {:db/ident        :pyg/LINKX,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def LabelPropagation
  "The label propagation operator from the `\"Learning from Labeled and\n    Unlabeled Datawith Label Propagation\"\n    <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper"
  {:db/ident        :pyg/LabelPropagation,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/MessagePassing})

(def LightGCN
  "The LightGCN model from the `\"LightGCN: Simplifying and Powering\n    Graph Convolution Network for Recommendation\"\n    <https://arxiv.org/abs/2002.02126>`_ paper."
  {:db/ident        :pyg/LightGCN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def MLP
  "A Multi-Layer Perception (MLP) model.\n    There exists two ways to instantiate an :class:`MLP`:"
  {:db/ident        :pyg/MLP,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def MaskLabel
  "The label embedding and masking layer from the `\"Masked Label\n    Prediction: Unified Message Passing Model for Semi-Supervised\n    Classification\" <https://arxiv.org/abs/2009.03509>`_ paper."
  {:db/ident        :pyg/MaskLabel,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def MetaPath2Vec
  "The MetaPath2Vec model from the `\"metapath2vec: Scalable Representation\n    Learning for Heterogeneous Networks\"\n    <https://ericdongyx.github.io/papers/\n    KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper where random walks based\n    on a given :obj:`metapath` are sampled in a heterogeneous graph, and node\n    embeddings are learned via negative sampling optimization."
  {:db/ident        :pyg/MetaPath2Vec,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def Node2Vec
  "The Node2Vec model from the\n    `\"node2vec: Scalable Feature Learning for Networks\"\n    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of\n    length :obj:`walk_length` are sampled in a given graph, and node embeddings\n    are learned via negative sampling optimization."
  {:db/ident        :pyg/Node2Vec,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def PNA
  "The Graph Neural Network from the `\"Principal Neighbourhood Aggregation\n    for Graph Nets\" <https://arxiv.org/abs/2004.05718>`_ paper, using the\n    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing."
  {:db/ident        :pyg/PNA,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/BasicGNN})

(def RECT_L
  "The RECT model, *i.e.* its supervised RECT-L part, from the\n    `\"Network Embedding with Completely-imbalanced Labels\"\n    <https://arxiv.org/abs/2007.03545>`_ paper.\n    In particular, a GCN model is trained that reconstructs semantic class\n    knowledge."
  {:db/ident        :pyg/RECT_L,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def RENet
  "The Recurrent Event Network model from the `\"Recurrent Event Network\n    for Reasoning over Temporal Knowledge Graphs\"\n    <https://arxiv.org/abs/1904.05530>`_ paper"
  {:db/ident        :pyg/RENet,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def SchNet
  "The continuous-filter convolutional neural network SchNet from the\n    `\"SchNet: A Continuous-filter Convolutional Neural Network for Modeling\n    Quantum Interactions\" <https://arxiv.org/abs/1706.08566>`_ paper that uses\n    the interactions blocks of the form"
  {:db/ident        :pyg/SchNet,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def SignedGCN
  "The signed graph convolutional network model from the `\"Signed Graph\n    Convolutional Network\" <https://arxiv.org/abs/1808.06354>`_ paper.\n    Internally, this module uses the\n    :class:`torch_geometric.nn.conv.SignedConv` operator."
  {:db/ident        :pyg/SignedGCN,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def TGNMemory
  "The Temporal Graph Network (TGN) memory model from the\n    `\"Temporal Graph Networks for Deep Learning on Dynamic Graphs\"\n    <https://arxiv.org/abs/2006.10637>`_ paper."
  {:db/ident        :pyg/TGNMemory,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :torch/Module})

(def VGAE
  "The Variational Graph Auto-Encoder model from the\n    `\"Variational Graph Auto-Encoders\" <https://arxiv.org/abs/1611.07308>`_\n    paper."
  {:db/ident        :pyg/VGAE,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg/GAE})

(def APPNP
  "The approximate personalized propagation of neural predictions layer from the “Predict then Propagate: Graph Neural Networks meet Personalized PageRank” paper."
  {:db/ident        :pyg/APPNP,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/MessagePassing
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.APPNP.html#torch_geometric.nn.conv.APPNP"
                     "https://arxiv.org/abs/1810.05997"]})

(def MetaLayer
  "A meta layer for building any kind of graph network, inspired by
  the “Relational Inductive Biases, Deep Learning, and Graph Networks”
  paper."
  {:db/ident        :pyg/MetaLayer,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :pyg/Module})

;; KGE Models

(def KGEModel
  "An abstract base class for implementing custom KGE models."
  {:db/ident        :pyg/KGEModel
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.KGEModel.html#torch_geometric.nn.kge.KGEModel"})

(def TransE
  "The TransE model from the “Translating Embeddings for Modeling Multi-Relational Data” paper."
  {:db/ident        :pyg/KGEModel
   :rdf/type        :owl/Class
   :rdfs/subClassOf :pyg/KGEModel
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.TransE.html#torch_geometric.nn.kge.TransE"
                     "https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf"]})

;; Encodings

(def PositionalEncoding
  "The positional encoding scheme from the “Attention Is All You Need” paper"
  {:db/ident        :pyg/PositionalEncoding,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    ["https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.encoding.PositionalEncoding"
                     "https://arxiv.org/pdf/1706.03762.pdf"]})

(def TemporalEncoding
  "The time-encoding function from the “Do We Really Need Complicated
  Model Architectures for Temporal Networks?” paper. TemporalEncoding
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
  {:db/ident        :pyg/DenseGCNConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    :pyg/GCNConv})

(def DenseGINConv
  {:db/ident        :pyg/DenseGINConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    :pyg/GINConv})

(def DenseGraphConv
  {:db/ident        :pyg/DenseGraphConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    :pyg/GraphConv})

(def DenseSAGEConv
  {:db/ident        :pyg/DenseSAGEConv,
   :rdf/type        :owl/Class   
   :rdfs/subClassOf :torch/Module
   :rdfs/seeAlso    :pyg/SAGEConv})

;; Dense Pooling Layers

(def DMoNPooling
  "The spectral modularity pooling operator from the “Graph Clustering with Graph Neural Networks” paper"
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
   :rdfs/subClassOf :py/Object})

;; DataParallel Layers

(def DataParallel
  {:db/ident        :pyg/DataParallel
   :rdf/type        :owl/Class
   :rdfs/subClassOf :torch/DataParallel})
