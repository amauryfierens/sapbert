"""
This script is used to create a graph from the snomed dataset.
"""

import random
import networkx as nx
from pathlib import Path
from itertools import chain
import os
from tqdm import tqdm
import pandas as pd

from datasets import Dataset

import torch_geometric as pyg

mrconso_structure = {
    "CUI": "Unique identifier for concept",
    "LAT": "Language of term",
    "TS": "Term status",
    "LUI": "Unique identifier for term",
    "STT": "String type",
    "SUI": "Unique identifier for string",
    "ISPREF": "Atom status - preferred (Y) or not (N) for this string within this concept",
    "AUI": "Unique identifier for atom - variable length field, 8 or 9 characters",
    "SAUI": "Source asserted atom identifier [optional]",
    "SCUI": "Source asserted concept identifier [optional]",
    "SDUI": "Source asserted descriptor identifier [optional]",
    "SAB": "Abbreviated source name (SAB). Maximum field length is 20 alphanumeric characters.",
    "TTY": "Abbreviation for term type in source vocabulary",
    "CODE": '''Most useful source asserted identifier (if the source vocabulary has more than one identifier), 
    or a Metathesaurus-generated source entry identifier (if the source vocabulary has none)''',
    "STR": "String",
    "SRL": "Source restriction level",
    "SUPPRESS": '''Suppressible flag. 
    Values = O (obsolete content), 
    E (Non-obsolete content marked suppressible by an editor), 
    Y (Non-obsolete content deemed suppressible during inversion), 
    or N (None of the above)''',
    "CVF": "Content View Flag. Bit field used to flag rows included in Content View.",
}


class SnomedGraph:
    """
    Class to create a graph from the snomed dataset.
    """
    def __init__(self,
                 snomed_int_path,
                 snomed_extension_path_list=[],
                 lang_list=["en", "fr"],
                 sep_token="<\s>",
                 add_umls_synonyms=False,
                 mrconso_path=None,
                 ):
        """
        Init the graph from the snomed dataset.
        :param snomed_int_path: Path to the snomed international distribution
        :param snomed_extension_path_list: List of path to the snomed extensions (if any, empty list otherwise)
        :param lang_list: List of languages to keep (default: ["en"])
        :param sep_token: Token used to separate descriptions from the same concept
        :param add_umls_synonyms: Whether to add UMLS synonyms to the graph
        :param mrconso_path: Path to the MRCONSO.RRF file (if add_umls_synonyms is True)
        """
        self.sep_token = sep_token
        self.lang_list = lang_list

        self.snomed_path_list = [
            Path(snomed_int_path) / "Snapshot/Terminology/",
        ]

        if snomed_extension_path_list is None:
            snomed_extension_path_list = []

        for ext_path in snomed_extension_path_list:
            self.snomed_path_list.append(Path(ext_path) / "Snapshot/Terminology/")

        for path in self.snomed_path_list:
            assert os.path.exists(path), "Path does not exists: " + str(path)

        self.add_umls_synonyms = add_umls_synonyms
        self.mrconso_path = mrconso_path

        if add_umls_synonyms:
            assert mrconso_path is not None, "mrconso_path must be provided if add_umls_synonyms is True"

        self.graph, self.desc_dataset = self.create_graph()
        self.graph_undir = self.graph.to_undirected()
        self.pyg_graph, self.concepts_mapping = self.pyg_graph()

    def create_graph(self):
        """
        Create the graph from the snomed dataset.
        :return: networkx graph
        """

        print("Creating graph...")

        # Create graph
        graph = nx.DiGraph()

        # Get active concepts
        df_list = []

        for file_path in chain(*[path.glob("sct2_Concept*") for path in self.snomed_path_list]):
            # Load file
            df_list.append(pd.read_csv(file_path,
                                       sep="\t",
                                       index_col=False,
                                       dtype={"id": int, "active": int},
                                       usecols=["id", "active"],
                                       low_memory=True,
                                      ))

        concepts_df = pd.concat(df_list)
        del df_list

        # Keep only active concepts
        concepts_df = concepts_df[concepts_df.active == 1]

        # Get set of concepts as nodes
        nodes = set(concepts_df.id.to_list())

        # Free df memory
        del concepts_df

        # Get definitions
        df_list = []

        for file_path in chain(*[path.glob("sct2_Description*") for path in self.snomed_path_list]):
            # filter by language (with "-en_" or "-fr_" in the filename)
            filter_list = ["-" + lang + "_" for lang in self.lang_list]
            if any(filter in str(file_path) for filter in filter_list):
                # Load file
                df_list.append(pd.read_csv(file_path,
                                           sep="\t",
                                           index_col=False,
                                           dtype={"conceptId": int, "term": str},
                                           usecols=["conceptId", "term"],
                                           low_memory=True,
                                           )
                               )

        desc_df = pd.concat(df_list)
        del df_list

        desc_df = desc_df.dropna(subset=["term"])

        # Load MRCONSO.RRF file and add synonyms to desc_df
        if self.add_umls_synonyms:
            umls_df = pd.read_csv(self.mrconso_path,
                                  sep="|",
                                  header=None,
                                  names=list(mrconso_structure.keys()),
                                  index_col=False,
                                  dtype={"CUI": str, "STR": str, "LAT": str, "SAB": str, "SCUI": str},
                                  usecols=["CUI", "STR", "LAT", "SAB", "SCUI"],
                                  low_memory=True,
                                  )

            lang_correspondance = {
                "fr": "FRE",
                "en": "ENG",
                "es": "SPA",
                "de": "GER",
                "it": "ITA",
                "nl": "DUT",
                "pt": "POR",
            }

            umls_lang = [lang_correspondance[lang] for lang in self.lang_list]

            # keep only synonyms in the selected languages
            umls_df = umls_df[umls_df.LAT.isin(umls_lang)]

            # Create a mapping from umls cui to snomed conceptId
            umls_snomed_mapping = umls_df[umls_df.SAB == "SNOMEDCT_US"][["CUI", "SCUI"]].drop_duplicates()
            umls_snomed_mapping = umls_snomed_mapping.set_index("CUI").to_dict()["SCUI"]

            # Convert UMLS cui to snomed conceptId
            umls_df["conceptId"] = umls_df.CUI.map(umls_snomed_mapping)

            # Keep only UMLS concepts that are in the snomed dataset
            umls_df = umls_df[umls_df.conceptId.notna()]

            # Keep only the columns we need
            umls_df = umls_df[["conceptId", "STR"]]

            # Rename columns
            umls_df = umls_df.rename(columns={"STR": "term"})

            # Convert conceptId to int
            umls_df["conceptId"] = umls_df.conceptId.astype(int)

            # Add UMLS synonyms to desc_df
            desc_df = pd.concat([desc_df, umls_df])

            # Free df memory
            del umls_df

        # Remove duplicates
        desc_df = desc_df.drop_duplicates()

        # Keep only descriptions from nodes
        desc_df = desc_df[desc_df.conceptId.isin(nodes)]

        # Remove non str terms
        desc_df = desc_df[desc_df.term.apply(lambda x: isinstance(x, str))]

        # Merge descriptions by conceptId delimited by a sep token
        desc_df = desc_df.groupby("conceptId").term.apply(lambda desc: self.sep_token.join(desc)).reset_index()

        # Set conceptId as id
        desc_df = desc_df[["conceptId", "term"]].rename(columns={"conceptId": "id"})

        # Create term dataset
        desc_dataset = Dataset.from_pandas(desc_df)

        for index, node_id, term in tqdm(desc_df.itertuples(), total=len(desc_df), desc="Add nodes"):
            graph.add_node(node_id, concept_id=node_id)

        # Get relationship
        df_list = []

        for file_path in chain(*[path.glob("sct2_Relationship_*") for path in self.snomed_path_list]):
            # Load file
            df_list.append(pd.read_csv(file_path,
                                       sep="\t",
                                       index_col=False,
                                       dtype={"active": int, "sourceId": int, "destinationId": int, "typeId": int},
                                       usecols=["active", "sourceId", "destinationId", "typeId"],
                                       low_memory=True,
                                       )
                           )

        rel_df = pd.concat(df_list)
        del df_list

        # Keep only active edges
        rel_df = rel_df[rel_df.active == 1][["sourceId", "destinationId", "typeId"]]

        # Keep only edges linked to nodes
        rel_df = rel_df[rel_df.sourceId.isin(nodes)]
        rel_df = rel_df[rel_df.destinationId.isin(nodes)]

        # Keep only edges of type "is a"
        rel_df = rel_df[rel_df.typeId == 116680003]
        
        # Add edges to graph
        for index, source_id, destination_id, type_id in tqdm(rel_df.itertuples(), total=len(rel_df), desc="Add edges"):
            graph.add_edge(destination_id, source_id, edge_type_id=type_id)

        print("Graph created.")

        return graph, desc_dataset

    def pyg_graph(self):
        """
        Convert the graph to a pytorch geometric graph.
        Also return a mapping from concept_id to pytorch geometric node index.
        :return: tuple (pyg_graph, concepts_mapping)
        """
        # Convert graph to pytorch geometric
        print("Converting graph to pytorch geometric...")
        pyg_graph = pyg.utils.from_networkx(self.graph)

        # Create mapping from concept_id to pytorch geometric node index
        concepts_mapping = {concept_id.item(): idx for idx, concept_id in enumerate(pyg_graph.concept_id)}

        print("Graph converted.")

        return pyg_graph, concepts_mapping

    def get_pyg_nodes_desc(self, nodes_id, random_term=False):
        nodes_terms = self.desc_dataset[nodes_id]['term']
        if random_term:
            nodes_terms = [random.choice(term_list.split(self.sep_token)) for term_list in nodes_terms]

        return nodes_terms

    def get_multiple_pyg_nodes_desc(self, nodes_id_list, max_terms=10, random_term=False, flatten=False):
        nodes_terms = self.desc_dataset[nodes_id_list]['term']
        if random_term:
            nodes_terms = [random.sample(term_list.split(self.sep_token),
                                         k=len(term_list.split(self.sep_token)))[:max_terms]
                           for term_list in nodes_terms]
        else:
            nodes_terms = [term_list.split(self.sep_token)[:max_terms] for term_list in nodes_terms]

        # limit the number of terms to max_terms
        nodes_terms = [term_list[:max_terms] for term_list in nodes_terms]

        labels = []
        for node_id, term_list in zip(nodes_id_list, nodes_terms):
            labels.append([node_id] * len(term_list))

        if flatten:
            nodes_terms = [term for term_list in nodes_terms for term in term_list]
            labels = [label for label_list in labels for label in label_list]

        return nodes_terms, labels

    def get_pyg_edges_desc(self, type_id_list, random_term=False):
        nodes_id = [self.concepts_mapping[type_id] for type_id in type_id_list]

        return self.get_pyg_nodes_desc(nodes_id, random_term=random_term)
