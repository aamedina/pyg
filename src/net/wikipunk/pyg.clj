(ns net.wikipunk.pyg
  (:require
   [com.stuartsierra.component :as com]
   [net.wikipunk.rdf.pyg]))

(defrecord PyG []
  com/Lifecycle
  (start [this]
    this)
  (stop [this]
    this))
