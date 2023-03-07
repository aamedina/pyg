(ns net.wikipunk.pyg.boot
  {:rdf/type :jsonld/Context})

(def py
  {:rdf/type    :rdfa/PrefixMapping
   :rdfa/uri    "https://wikipunk.net/py/"
   :rdfa/prefix "py"})

(def torch
  {:rdf/type    :rdfa/PrefixMapping
   :rdfa/uri    "https://wikipunk.net/torch/"
   :rdfa/prefix "torch"})

(def pyg
  {:rdf/type    :rdfa/PrefixMapping
   :rdfa/uri    "https://wikipunk.net/pyg/"
   :rdfa/prefix "pyg"})
