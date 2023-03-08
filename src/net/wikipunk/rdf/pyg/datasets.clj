(ns net.wikipunk.rdf.pyg.datasets
  "RDF vocabulary for PyG datasets"
  {:rdf/type :owl/Ontology})

(def GraphGenerator
  "An abstract base class for generating synthetic graphs."
  {:db/ident        :pyg.datasets/GraphGenerator
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def BAGraph
  "Generates random Barabasi-Albert (BA) graphs."
  {:db/ident        :pyg.datasets/BAGraph
   :rdf/type        :owl/Class
   :rdfs/subClassOf :pyg.datasets/GraphGenerator})

(def ERGraph
  "Generates random Erdos-Renyi (ER) graphs."
  {:db/ident        :pyg.datasets/ERGraph
   :rdf/type        :owl/Class
   :rdfs/subClassOf :pyg.datasets/GraphGenerator})

(def GridGraph
  "Generates two-dimensional grid graphs."
  {:db/ident        :pyg.datasets/GridGraph
   :rdf/type        :owl/Class
   :rdfs/subClassOf :pyg.datasets/GraphGenerator})

(def AMiner
  "The heterogeneous AMiner dataset from the `\"metapath2vec: Scalable    Representation Learning for Heterogeneous Networks\"    <https://ericdongyx.github.io/papers/D17-dong-chawla-swami-metapath2vec.pdf>` paper, consisting of nodes from    type :`\"paper\"`, :`\"author\"` and :`\"venue\"`.    Venue categories and author research interests are available as ground    truth labels for a subset of nodes."
  {:db/ident :pyg.datasets/AMiner,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def AQSOL
  "The AQSOL dataset from the `Benchmarking Graph Neural Networks    <http://arxiv.org/abs/2003.00982>` paper based on    `AqSolDB <https://www.nature.com/articles/s41597-019-0151-1>`, a    standardized database of 9,982 molecular graphs with their aqueous    solubility values, collected from 9 different data sources."
  {:db/ident :pyg.datasets/AQSOL,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Actor
  "The actor-only induced subgraph of the film-director-actor-writer    network used in the    `\"Geom-GCN: Geometric Graph Convolutional Networks\"    <https://openreview.net/forum?id=S1e2agrFvS>` paper.    Each node corresponds to an actor, and the edge between two nodes denotes    co-occurrence on the same Wikipedia page.    Node features correspond to some keywords in the Wikipedia pages.    The task is to classify the nodes into five categories in term of words of    actor's Wikipedia."
  {:db/ident :pyg.datasets/Actor,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Airports
  "The Airports dataset from the `\"struc2vec: Learning Node    Representations from Structural Identity\"    <https://arxiv.org/abs/1704.03165>` paper, where nodes denote airports    and labels correspond to activity levels.    Features are given by one-hot encoded node identifiers, as described in the    `\"GraLSP: Graph Neural Networks with Local Structural Patterns\"    ` <https://arxiv.org/abs/1911.07675>` paper."
  {:db/ident :pyg.datasets/Airports,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Amazon
  "The Amazon Computers and Amazon Photo networks from the    `\"Pitfalls of Graph Neural Network Evaluation\"    <https://arxiv.org/abs/1811.05868>` paper.    Nodes represent goods and edges represent that two goods are frequently    bought together.    Given product reviews as bag-of-words node features, the task is to    map goods to their respective product category."
  {:db/ident :pyg.datasets/Amazon,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def AmazonProducts
  "The Amazon dataset from the `\"GraphSAINT: Graph Sampling Based    Inductive Learning Method\" <https://arxiv.org/abs/1907.04931>` paper,    containing products and its categories."
  {:db/ident :pyg.datasets/AmazonProducts,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def AttributedGraphDataset
  "A variety of attributed graph datasets from the    `\"Scaling Attributed Network Embedding to Massive Graphs\"    <https://arxiv.org/abs/2009.00826>` paper."
  {:db/ident :pyg.datasets/AttributedGraphDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def BAShapes
  "The BA-Shapes dataset from the `\"GNNExplainer: Generating Explanations    for Graph Neural Networks\" <https://arxiv.org/pdf/1903.03894.pdf>` paper,    containing a Barabasi-Albert (BA) graph with 300 nodes and a set of 80    \"house\"-structured graphs connected to it."
  {:db/ident :pyg.datasets/BAShapes,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def BitcoinOTC
  "The Bitcoin-OTC dataset from the `\"EvolveGCN: Evolving Graph    Convolutional Networks for Dynamic Graphs\"    <https://arxiv.org/abs/1902.10191>` paper, consisting of 138    who-trusts-whom networks of sequential time steps."
  {:db/ident :pyg.datasets/BitcoinOTC,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def CitationFull
  "The full citation network datasets from the    `\"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via    Ranking\" <https://arxiv.org/abs/1707.03815>` paper.    Nodes represent documents and edges represent citation links.    Datasets include `citeseer`, `cora`, `coraml`, `dblp`, `pubmed`."
  {:db/ident :pyg.datasets/CitationFull,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def CoMA
  "The CoMA 3D faces dataset from the `\"Generating 3D faces using    Convolutional Mesh Autoencoders\" <https://arxiv.org/abs/1807.10267>`    paper, containing 20,466 meshes of extreme expressions captured over 12    different subjects."
  {:db/ident :pyg.datasets/CoMA,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Coauthor
  "The Coauthor CS and Coauthor Physics networks from the    `\"Pitfalls of Graph Neural Network Evaluation\"    <https://arxiv.org/abs/1811.05868>` paper.    Nodes represent authors that are connected by an edge if they co-authored a    paper.    Given paper keywords for each author's papers, the task is to map authors    to their respective field of study."
  {:db/ident :pyg.datasets/Coauthor,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def CoraFull
  "Alias for :`torchgeometric.datasets.CitationFull` with    :`name=\"cora\"`."
  {:db/ident        :pyg.datasets/CoraFull,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg.datasets/CitationFull})

(def DBLP
  "A subset of the DBLP computer science bibliography website, as    collected in the `\"MAGNN: Metapath Aggregated Graph Neural Network for    Heterogeneous Graph Embedding\" <https://arxiv.org/abs/2002.01680>` paper.    DBLP is a heterogeneous graph containing four types of entities - authors    (4,057 nodes), papers (14,328 nodes), terms (7,723 nodes), and conferences    (20 nodes).    The authors are divided into four research areas (database, data mining,    artificial intelligence, information retrieval).    Each author is described by a bag-of-words representation of their paper    keywords."
  {:db/ident :pyg.datasets/DBLP,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def DBP15K
  "The DBP15K dataset from the    `\"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding\"    <https://arxiv.org/abs/1708.05045>` paper, where Chinese, Japanese and    French versions of DBpedia were linked to its English version.    Node features are given by pre-trained and aligned monolingual word    embeddings from the `\"Cross-lingual Knowledge Graph Alignment via Graph    Matching Neural Network\" <https://arxiv.org/abs/1905.11605>` paper."
  {:db/ident :pyg.datasets/DBP15K,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def DGraphFin
  "The DGraphFin networks from the    `\"DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection\"    <https://arxiv.org/abs/2207.03579>` paper.    It is a directed, unweighted dynamic graph consisting of millions of    nodes and edges, representing a realistic user-to-user social network    in financial industry.    Node represents a Finvolution user, and an edge from one    user to another means that the user regards the other user    as the emergency contact person. Each edge is associated with a    timestamp ranging from 1 to 821 and a type of emergency contact    ranging from 0 to 11."
  {:db/ident :pyg.datasets/DGraphFin,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def DeezerEurope
  "The Deezer Europe dataset introduced in the `\"Characteristic Functions    on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric    Models\" <https://arxiv.org/abs/2005.07959>` paper.    Nodes represent European users of Deezer and edges are mutual follower    relationships.    It contains 28,281 nodes, 185,504 edges, 128 node features and 2 classes."
  {:db/ident :pyg.datasets/DeezerEurope,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def DynamicFAUST
  "The dynamic FAUST humans dataset from the `\"Dynamic FAUST: Registering    Human Bodies in Motion\"    <http://files.is.tue.mpg.de/black/papers/dfaust2017.pdf>` paper."
  {:db/ident :pyg.datasets/DynamicFAUST,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def EllipticBitcoinDataset
  "The Elliptic Bitcoin dataset of Bitcoin transactions from the    `\"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional    Networks for Financial Forensics\" <https://arxiv.org/abs/1908.02591>`    paper."
  {:db/ident :pyg.datasets/EllipticBitcoinDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def EmailEUCore
  "An e-mail communication network of a large European research    institution, taken from the `\"Local Higher-order Graph Clustering\"    <https://www-cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf>` paper.    Nodes indicate members of the institution.    An edge between a pair of members indicates that they exchanged at least    one email.    Node labels indicate membership to one of the 42 departments."
  {:db/ident :pyg.datasets/EmailEUCore,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Entities
  "The relational entities networks \"AIFB\", \"MUTAG\", \"BGS\" and \"AM\" from    the `\"Modeling Relational Data with Graph Convolutional Networks\"    <https://arxiv.org/abs/1703.06103>` paper.    Training and test splits are given by node indices."
  {:db/ident :pyg.datasets/Entities,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def FAUST
  "The FAUST humans dataset from the `\"FAUST: Dataset and Evaluation for    3D Mesh Registration\"    <http://files.is.tue.mpg.de/black/papers/FAUST2014.pdf>` paper,    containing 100 watertight meshes representing 10 different poses for 10    different subjects."
  {:db/ident :pyg.datasets/FAUST,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def FacebookPagePage
  "The Facebook Page-Page network dataset introduced in the    `\"Multi-scale Attributed Node Embedding\"    <https://arxiv.org/abs/1909.13021>` paper.    Nodes represent verified pages on Facebook and edges are mutual likes.    It contains 22,470 nodes, 342,004 edges, 128 node features and 4 classes."
  {:db/ident :pyg.datasets/FacebookPagePage,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def FakeDataset
  "A fake dataset that returns randomly generated    :`~torchgeometric.data.Data` objects."
  {:db/ident :pyg.datasets/FakeDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def FakeHeteroDataset
  "A fake dataset that returns randomly generated    :`~torchgeometric.data.HeteroData` objects."
  {:db/ident :pyg.datasets/FakeHeteroDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Flickr
  "The Flickr dataset from the `\"GraphSAINT: Graph Sampling Based    Inductive Learning Method\" <https://arxiv.org/abs/1907.04931>` paper,    containing descriptions and common properties of images."
  {:db/ident :pyg.datasets/Flickr,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def GDELT
  "The Global Database of Events, Language, and Tone (GDELT) dataset used    in the, *e.g.*, `\"Recurrent Event Network for Reasoning over Temporal    Knowledge Graphs\" <https://arxiv.org/abs/1904.05530>` paper, consisting of    events collected from 1/1/2018 to 1/31/2018 (15 minutes time granularity)."
  {:db/ident :pyg.datasets/GDELT,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def GEDDataset
  "The GED datasets from the `\"Graph Edit Distance Computation via Graph    Neural Networks\" <https://arxiv.org/abs/1808.05689>` paper.    GEDs can be accessed via the global attributes :`ged` and    :`normged` for all train/train graph pairs and all train/test graph    pairs:"
  {:db/ident :pyg.datasets/GEDDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def GNNBenchmarkDataset
  "A variety of artificially and semi-artificially generated graph    datasets from the `\"Benchmarking Graph Neural Networks\"    <https://arxiv.org/abs/2003.00982>` paper."
  {:db/ident :pyg.datasets/GNNBenchmarkDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def GemsecDeezer
  "The Deezer User Network datasets introduced in the    `\"GEMSEC: Graph Embedding with Self Clustering\"    <https://arxiv.org/abs/1802.03997>` paper.    Nodes represent Deezer user and edges are mutual friendships.    The task is multi-label multi-class node classification about    the genres liked by the users."
  {:db/ident :pyg.datasets/GemsecDeezer,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def GeometricShapes
  "Synthetic dataset of various geometric shapes like cubes, spheres or    pyramids."
  {:db/ident :pyg.datasets/GeometricShapes,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def GitHub
  "The GitHub Web and ML Developers dataset introduced in the    `\"Multi-scale Attributed Node Embedding\"    <https://arxiv.org/abs/1909.13021>` paper.    Nodes represent developers on GitHub and edges are mutual follower    relationships.    It contains 37,300 nodes, 578,006 edges, 128 node features and 2 classes."
  {:db/ident :pyg.datasets/GitHub,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def HGBDataset
  "A variety of heterogeneous graph benchmark datasets from the    `\"Are We Really Making Much Progress? Revisiting, Benchmarking, and    Refining Heterogeneous Graph Neural Networks\"    <http://keg.cs.tsinghua.edu.cn/jietang/publications/    KDD21-Lv-et-al-HeterGNN.pdf>` paper."
  {:db/ident :pyg.datasets/HGBDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def HydroNet
  "The HydroNet dataest from the    `\"HydroNet: Benchmark Tasks for Preserving Intermolecular Interactions and    Structural Motifs in Predictive and Generative Models for Molecular Data\"    <https://arxiv.org/abs/2012.00131>` paper, consisting of 5 million water    clusters held together by hydrogen bonding networks.  This dataset    provides atomic coordinates and total energy in kcal/mol for the cluster."
  {:db/ident :pyg.datasets/HydroNet,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def ICEWS18
  "The Integrated Crisis Early Warning System (ICEWS) dataset used in    the, *e.g.*, `\"Recurrent Event Network for Reasoning over Temporal    Knowledge Graphs\" <https://arxiv.org/abs/1904.05530>` paper, consisting of    events collected from 1/1/2018 to 10/31/2018 (24 hours time granularity)."
  {:db/ident :pyg.datasets/ICEWS18,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def IMDB
  "A subset of the Internet Movie Database (IMDB), as collected in the    `\"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph    Embedding\" <https://arxiv.org/abs/2002.01680>` paper.    IMDB is a heterogeneous graph containing three types of entities - movies    (4,278 nodes), actors (5,257 nodes), and directors (2,081 nodes).    The movies are divided into three classes (action, comedy, drama) according    to their genre.    Movie features correspond to elements of a bag-of-words representation of    its plot keywords."
  {:db/ident :pyg.datasets/IMDB,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def JODIEDataset
  "JODIEDataset"
  {:db/ident :pyg.datasets/JODIEDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def KarateClub
  "Zachary's karate club network from the `\"An Information Flow Model for    Conflict and Fission in Small Groups\"    <http://www1.ind.ku.dk/complexLearning/zachary1977.pdf>` paper, containing    34 nodes, connected by 156 (undirected and unweighted) edges.    Every node is labeled by one of four classes obtained via modularity-based    clustering, following the `\"Semi-supervised Classification with Graph    Convolutional Networks\" <https://arxiv.org/abs/1609.02907>` paper.    Training is based on a single labeled example per class, *i.e.* a total    number of 4 labeled nodes."
  {:db/ident :pyg.datasets/KarateClub,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def LINKXDataset
  "A variety of non-homophilous graph datasets from the `\"Large Scale    Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple    Methods\" <https://arxiv.org/abs/2110.14446>` paper."
  {:db/ident :pyg.datasets/LINKXDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def LRGBDataset
  "The `\"Long Range Graph Benchmark (LRGB)\"    <https://arxiv.org/abs/2206.08164>`    datasets which is a collection of 5 graph learning datasets with tasks    that are based on long-range dependencies in graphs. See the original    `source code <https://github.com/vijaydwivedi75/lrgb>` for more details    on the individual datasets."
  {:db/ident :pyg.datasets/LRGBDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def LastFM
  "A subset of the last.fm music website keeping track of users' listining    information from various sources, as collected in the    `\"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph    Embedding\" <https://arxiv.org/abs/2002.01680>` paper.    last.fm is a heterogeneous graph containing three types of entities - users    (1,892 nodes), artists (17,632 nodes), and artist tags (1,088 nodes).    This dataset can be used for link prediction, and no labels or features are    provided."
  {:db/ident :pyg.datasets/LastFM,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def LastFMAsia
  "The LastFM Asia Network dataset introduced in the `\"Characteristic    Functions on Graphs: Birds of a Feather, from Statistical Descriptors to    Parametric Models\" <https://arxiv.org/abs/2005.07959>` paper.    Nodes represent LastFM users from Asia and edges are friendships.    It contains 7,624 nodes, 55,612 edges, 128 node features and 18 classes."
  {:db/ident :pyg.datasets/LastFMAsia,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def MD17
  "A variety of ab-initio molecular dynamics trajectories from the    authors of `sGDML <http://quantum-machine.org/gdml>`.    This class provides access to the original MD17 datasets as well as all    other datasets released by sGDML since then (15 in total)."
  {:db/ident :pyg.datasets/MD17,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def MNISTSuperpixels
  "MNIST superpixels dataset from the `\"Geometric Deep Learning on    Graphs and Manifolds Using Mixture Model CNNs\"    <https://arxiv.org/abs/1611.08402>` paper, containing 70,000 graphs with    75 nodes each.    Every graph is labeled by one of 10 classes."
  {:db/ident :pyg.datasets/MNISTSuperpixels,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def MalNetTiny
  "The MalNet Tiny dataset from the    `\"A Large-Scale Database for Graph Representation Learning\"    <https://openreview.net/pdf?id=1xDTDk3XPW>` paper.    :`MalNetTiny` contains 5,000 malicious and benign software function    call graphs across 5 different types. Each graph contains at most 5k nodes."
  {:db/ident :pyg.datasets/MalNetTiny,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def MixHopSyntheticDataset
  "The MixHop synthetic dataset from the `\"MixHop: Higher-Order    Graph Convolutional Architectures via Sparsified Neighborhood Mixing\"    <https://arxiv.org/abs/1905.00067>` paper, containing 10    graphs, each with varying degree of homophily (ranging from 0.0 to 0.9).    All graphs have 5,000 nodes, where each node corresponds to 1 out of 10    classes.    The feature values of the nodes are sampled from a 2D Gaussian    distribution, which are distinct for each class."
  {:db/ident :pyg.datasets/MixHopSyntheticDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def ModelNet
  "The ModelNet10/40 datasets from the `\"3D ShapeNets: A Deep    Representation for Volumetric Shapes\"    <https://people.csail.mit.edu/khosla/papers/cvpr2015wu.pdf>` paper,    containing CAD models of 10 and 40 categories, respectively."
  {:db/ident :pyg.datasets/ModelNet,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def MoleculeNet
  "The `MoleculeNet <http://moleculenet.org/datasets-1>` benchmark    collection  from the `\"MoleculeNet: A Benchmark for Molecular Machine    Learning\" <https://arxiv.org/abs/1703.00564>` paper, containing datasets    from physical chemistry, biophysics and physiology.    All datasets come with the additional node and edge features introduced by    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`."
  {:db/ident :pyg.datasets/MoleculeNet,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def MovieLens
  "A heterogeneous rating dataset, assembled by GroupLens Research from    the `MovieLens web site <https://movielens.org>`, consisting of nodes of    type :`\"movie\"` and :`\"user\"`.    User ratings for movies are available as ground truth labels for the edges    between the users and the movies :`(\"user\", \"rates\", \"movie\")`."
  {:db/ident :pyg.datasets/MovieLens,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def NELL
  "The NELL dataset, a knowledge graph from the    `\"Toward an Architecture for Never-Ending Language Learning\"    <https://www.cs.cmu.edu/~acarlson/papers/carlson-aaai10.pdf>` paper.    The dataset is processed as in the    `\"Revisiting Semi-Supervised Learning with Graph Embeddings\"    <https://arxiv.org/abs/1603.08861>` paper."
  {:db/ident :pyg.datasets/NELL,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def OGBMAG
  "The ogbn-mag dataset from the `\"Open Graph Benchmark: Datasets for    Machine Learning on Graphs\" <https://arxiv.org/abs/2005.00687>` paper.    ogbn-mag is a heterogeneous graph composed of a subset of the Microsoft    Academic Graph (MAG).    It contains four types of entities — papers (736,389 nodes), authors    (1,134,649 nodes), institutions (8,740 nodes), and fields of study    (59,965 nodes) — as well as four types of directed relations connecting two    types of entities.    Each paper is associated with a 128-dimensional :`word2vec` feature    vector, while all other node types are not associated with any input    features.    The task is to predict the venue (conference or journal) of each paper.    In total, there are 349 different venues."
  {:db/ident :pyg.datasets/OGBMAG,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def OMDB
  "The `Organic Materials Database (OMDB)    <https://omdb.mathub.io/dataset>` of bulk organic crystals."
  {:db/ident :pyg.datasets/OMDB,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def PCPNetDataset
  "The PCPNet dataset from the `\"PCPNet: Learning Local Shape Properties    from Raw Point Clouds\" <https://arxiv.org/abs/1710.04954>` paper,    consisting of 30 shapes, each given as a point cloud, densely sampled with    100k points.    For each shape, surface normals and local curvatures are given as node    features."
  {:db/ident :pyg.datasets/PCPNetDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def PPI
  "The protein-protein interaction networks from the `\"Predicting    Multicellular Function through Multi-layer Tissue Networks\"    <https://arxiv.org/abs/1707.04638>` paper, containing positional gene    sets, motif gene sets and immunological signatures as features (50 in    total) and gene ontology sets as labels (121 in total)."
  {:db/ident :pyg.datasets/PPI,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def PascalPF
  "The Pascal-PF dataset from the `\"Proposal Flow\"    <https://arxiv.org/abs/1511.05065>` paper, containing 4 to 16 keypoints    per example over 20 categories."
  {:db/ident :pyg.datasets/PascalPF,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def PascalVOCKeypoints
  "The Pascal VOC 2011 dataset with Berkely annotations of keypoints from    the `\"Poselets: Body Part Detectors Trained Using 3D Human Pose    Annotations\" <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/    human/ poseletsiccv09.pdf>` paper, containing 0 to 23 keypoints per    example over 20 categories.    The dataset is pre-filtered to exclude difficult, occluded and truncated    objects.    The keypoints contain interpolated features from a pre-trained VGG16 model    on ImageNet (:`relu42` and :`relu51`)."
  {:db/ident :pyg.datasets/PascalVOCKeypoints,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Planetoid
  "The citation network datasets \"Cora\", \"CiteSeer\" and \"PubMed\" from the    `\"Revisiting Semi-Supervised Learning with Graph Embeddings\"    <https://arxiv.org/abs/1603.08861>` paper.    Nodes represent documents and edges represent citation links.    Training, validation and test splits are given by binary masks."
  {:db/ident :pyg.datasets/Planetoid,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def PolBlogs
  "The Political Blogs dataset from the `\"The Political Blogosphere and    the 2004 US Election: Divided they Blog\"    <https://dl.acm.org/doi/10.1145/1134271.1134277>` paper."
  {:db/ident :pyg.datasets/PolBlogs,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def QM7b
  "The QM7b dataset from the `\"MoleculeNet: A Benchmark for Molecular    Machine Learning\" <https://arxiv.org/abs/1703.00564>` paper, consisting of    7,211 molecules with 14 regression targets."
  {:db/ident :pyg.datasets/QM7b,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def QM9
  "The QM9 dataset from the `\"MoleculeNet: A Benchmark for Molecular    Machine Learning\" <https://arxiv.org/abs/1703.00564>` paper, consisting of    about 130,000 molecules with 19 regression targets.    Each molecule includes complete spatial information for the single low    energy conformation of the atoms in the molecule.    In addition, we provide the atom features from the `\"Neural Message    Passing for Quantum Chemistry\" <https://arxiv.org/abs/1704.01212>` paper."
  {:db/ident :pyg.datasets/QM9,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def RandomPartitionGraphDataset
  "The random partition graph dataset from the `\"How to Find Your    Friendly Neighborhood: Graph Attention Design with Self-Supervision\"    <https://openreview.net/forum?id=Wi5KUNlqWty>` paper.    This is a synthetic graph of communities controlled by the node homophily    and the average degree, and each community is considered as a class.    The node features are sampled from normal distributions where the centers    of clusters are vertices of a hypercube, as computed by the    :meth:`sklearn.datasets.makeclassification` method."
  {:db/ident        :pyg.datasets/RandomPartitionGraphDataset,
   :rdf/type        :owl/Class,
   :rdfs/subClassOf :pyg.datasets/StochasticBlockModelDataset})

(def Reddit
  "The Reddit dataset from the `\"Inductive Representation Learning on    Large Graphs\" <https://arxiv.org/abs/1706.02216>` paper, containing    Reddit posts belonging to different communities."
  {:db/ident :pyg.datasets/Reddit,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Reddit2
  "The Reddit dataset from the `\"GraphSAINT: Graph Sampling Based    Inductive Learning Method\" <https://arxiv.org/abs/1907.04931>` paper,    containing Reddit posts belonging to different communities."
  {:db/ident :pyg.datasets/Reddit2,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def RelLinkPredDataset
  "The relational link prediction datasets from the    `\"Modeling Relational Data with Graph Convolutional Networks\"    <https://arxiv.org/abs/1703.06103>` paper.    Training and test splits are given by sets of triplets."
  {:db/ident :pyg.datasets/RelLinkPredDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def S3DIS
  "The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from    the `\"3D Semantic Parsing of Large-Scale Indoor Spaces\"    <https://openaccess.thecvf.com/contentcvpr2016/papers/Armeni3DSemanticParsingCVPR2016paper.pdf>`    paper, containing point clouds of six large-scale indoor parts in three    buildings with 12 semantic elements (and one clutter class)."
  {:db/ident :pyg.datasets/S3DIS,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def SHREC2016
  "The SHREC 2016 partial matching dataset from the `\"SHREC'16: Partial    Matching of Deformable Shapes\"    <http://www.dais.unive.it/~shrec2016/shrec16-partial.pdf>` paper.    The reference shape can be referenced via :`dataset.ref`."
  {:db/ident :pyg.datasets/SHREC2016,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def SNAPDataset
  "A variety of graph datasets collected from `SNAP at Stanford University    <https://snap.stanford.edu/data>`."
  {:db/ident :pyg.datasets/SNAPDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def ShapeNet
  "The ShapeNet part level segmentation dataset from the `\"A Scalable    Active Framework for Region Annotation in 3D Shape Collections\"    <http://web.stanford.edu/~ericyi/papers/partannotation16small.pdf>`    paper, containing about 17,000 3D shape point clouds from 16 shape    categories.    Each category is annotated with 2 to 6 parts."
  {:db/ident :pyg.datasets/ShapeNet,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def StochasticBlockModelDataset
  "A synthetic graph dataset generated by the stochastic block model.    The node features of each block are sampled from normal distributions where    the centers of clusters are vertices of a hypercube, as computed by the    :meth:`sklearn.datasets.makeclassification` method."
  {:db/ident :pyg.datasets/StochasticBlockModelDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def SuiteSparseMatrixCollection
  "A suite of sparse matrix benchmarks known as the `Suite Sparse Matrix    Collection <https://sparse.tamu.edu>` collected from a wide range of    applications."
  {:db/ident :pyg.datasets/SuiteSparseMatrixCollection,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def TOSCA
  "The TOSCA dataset from the `\"Numerical Geometry of Non-Ridig Shapes\"    <https://www.amazon.com/Numerical-Geometry-Non-Rigid-Monographs-Computer/    dp/0387733000>` book, containing 80 meshes.    Meshes within the same category have the same triangulation and an equal    number of vertices numbered in a compatible way."
  {:db/ident :pyg.datasets/TOSCA,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def TUDataset
  "A variety of graph kernel benchmark datasets, *.e.g.* \"IMDB-BINARY\",    \"REDDIT-BINARY\" or \"PROTEINS\", collected from the `TU Dortmund University    <https://chrsmrrs.github.io/datasets>`.    In addition, this dataset wrapper provides `cleaned dataset versions    <https://github.com/nd7141/graphdatasets>` as motivated by the    `\"Understanding Isomorphism Bias in Graph Data Sets\"    <https://arxiv.org/abs/1910.12091>` paper, containing only non-isomorphic    graphs."
  {:db/ident :pyg.datasets/TUDataset,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Twitch
  "The Twitch Gamer networks introduced in the    `\"Multi-scale Attributed Node Embedding\"    <https://arxiv.org/abs/1909.13021>` paper.    Nodes represent gamers on Twitch and edges are followerships between them.    Node features represent embeddings of games played by the Twitch users.    The task is to predict whether a user streams mature content."
  {:db/ident :pyg.datasets/Twitch,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def UPFD
  "The tree-structured fake news propagation graph classification dataset    from the `\"User Preference-aware Fake News Detection\"    <https://arxiv.org/abs/2104.12259>` paper.    It includes two sets of tree-structured fake & real news propagation graphs    extracted from Twitter.    For a single graph, the root node represents the source news, and leaf    nodes represent Twitter users who retweeted the same root news.    A user node has an edge to the news node if and only if the user retweeted    the root news directly.    Two user nodes have an edge if and only if one user retweeted the root news    from the other user.    Four different node features are encoded using different encoders.    Please refer to `GNN-FakeNews    <https://github.com/safe-graph/GNN-FakeNews>` repo for more details."
  {:db/ident :pyg.datasets/UPFD,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def WILLOWObjectClass
  "The WILLOW-ObjectClass dataset from the `\"Learning Graphs to Match\"    <https://www.di.ens.fr/willow/pdfscurrent/cho2013.pdf>` paper,    containing 10 equal keypoints of at least 40 images in each category.    The keypoints contain interpolated features from a pre-trained VGG16 model    on ImageNet (:`relu42` and :`relu51`)."
  {:db/ident :pyg.datasets/WILLOWObjectClass,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def WebKB
  "The WebKB datasets used in the    `\"Geom-GCN: Geometric Graph Convolutional Networks\"    <https://openreview.net/forum?id=S1e2agrFvS>` paper.    Nodes represent web pages and edges represent hyperlinks between them.    Node features are the bag-of-words representation of web pages.    The task is to classify the nodes into one of the five categories, student,    project, course, staff, and faculty."
  {:db/ident :pyg.datasets/WebKB,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def WikiCS
  "The semi-supervised Wikipedia-based dataset from the    `\"Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks\"    <https://arxiv.org/abs/2007.02901>` paper, containing 11,701 nodes,    216,123 edges, 10 classes and 20 different training splits."
  {:db/ident :pyg.datasets/WikiCS,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def WikipediaNetwork
  "The Wikipedia networks introduced in the    `\"Multi-scale Attributed Node Embedding\"    <https://arxiv.org/abs/1909.13021>` paper.    Nodes represent web pages and edges represent hyperlinks between them.    Node features represent several informative nouns in the Wikipedia pages.    The task is to predict the average daily traffic of the web page."
  {:db/ident :pyg.datasets/WikipediaNetwork,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def WordNet18
  "The WordNet18 dataset from the `\"Translating Embeddings for Modeling    Multi-Relational Data\"    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling    -multi-relational-data>` paper,    containing 40,943 entities, 18 relations and 151,442 fact triplets,    *e.g.*, furniture includes bed."
  {:db/ident :pyg.datasets/WordNet18,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def WordNet18RR
  "The WordNet18RR dataset from the `\"Convolutional 2D Knowledge Graph    Embeddings\" <https://arxiv.org/abs/1707.01476>` paper, containing 40,943    entities, 11 relations and 93,003 fact triplets."
  {:db/ident :pyg.datasets/WordNet18RR,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def Yelp
  "The Yelp dataset from the `\"GraphSAINT: Graph Sampling Based    Inductive Learning Method\" <https://arxiv.org/abs/1907.04931>` paper,    containing customer reviewers and their friendship."
  {:db/ident :pyg.datasets/Yelp,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})

(def ZINC
  "The ZINC dataset from the `ZINC database    <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>` and the    `\"Automatic Chemical Design Using a Data-Driven Continuous Representation    of Molecules\" <https://arxiv.org/abs/1610.02415>` paper, containing about    250,000 molecular graphs with up to 38 heavy atoms.    The task is to regress the penalized :`logP` (also called constrained    solubility in some works), given by :`y = logP - SAS - cycles`, where    :`logP` is the water-octanol partition coefficient, :`SAS` is the    synthetic accessibility score, and :`cycles` denotes the number of    cycles with more than six atoms.    Penalized :`logP` is a score commonly used for training molecular    generation models, see, *e.g.*, the    `\"Junction Tree Variational Autoencoder for Molecular Graph Generation\"    <https://proceedings.mlr.press/v80/jin18a.html>` and    `\"Grammar Variational Autoencoder\"    <https://proceedings.mlr.press/v70/kusner17a.html>` papers."
  {:db/ident :pyg.datasets/ZINC,
   :rdf/type [:pyg/InMemoryDataset :owl/NamedIndividual]})
