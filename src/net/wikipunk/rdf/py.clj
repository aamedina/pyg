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

(def Enum
  "The class of Python enums."
  {:db/ident        :py/Enum
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def ExceptionClass
  "The class of Python exceptions."
  {:db/ident        :py/Exception
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def Function
  "The class of Python functions."
  {:db/ident        :py/Function
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})
