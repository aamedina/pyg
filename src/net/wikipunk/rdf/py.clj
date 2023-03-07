(ns net.wikipunk.rdf.py
  "RDF vocabulary for Python"
  {:rdf/type :owl/Ontology})

(def T
  "The class of Python classes."
  {:db/ident :py/Class
   :rdf/type :owl/Class})

(def ObjectClass
  "The class of Python objects.

  Python objects are maps of attributes and items with a class."
  {:db/ident        :py/Object
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Class})
