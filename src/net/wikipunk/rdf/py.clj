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

(def EnumClass
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

(def Sized
  "The class of Sized collections."
  {:db/ident        :py/Sized
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def Iterable
  "An iterable object is an object that implements __iter__, which is
  expected to return an iterator object. "
  {:db/ident        :py/Iterable
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})

(def Iterator
  "An iterator object implements __next__, which is expected to return
  the next element of the iterable object that returned it, and to
  raise a StopIteration exception when no more elements are
  available."
  {:db/ident        :py/Iterator
   :rdf/type        :owl/Class
   :rdfs/subClassOf :py/Object})
