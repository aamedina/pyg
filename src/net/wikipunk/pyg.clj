(ns net.wikipunk.pyg
  (:require
   [com.stuartsierra.component :as com]
   [net.wikipunk.pyg.boot]))

(defrecord PyG []
  com/Lifecycle
  (start [this]
    this)
  (stop [this]
    this))
