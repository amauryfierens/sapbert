import networkx as nx
import os


class ICD10:
    def __init__(self, icd10_path):
        self.icd10_path = icd10_path
        self.definition_index = {}
        self.index_definition = {}
        self.graph = None
        
    def load_icd10(self):
        
        # init graph
        self.graph = nx.DiGraph()
        
        # codes for edge creation
        icd_codes = []
        
        # load definitions
        with open(self.icd10_path,
                  mode='r',
                  encoding='utf8') as f:
            
            for line in f:
                icd_code = line[6:13].strip()   # ICD10-CM and ICD10-PCS codes
                icd_codes.append(icd_code)      # for edges
                sdesc = line[16:76].strip()     # short description
                ldesc = line[77:].strip()       # long description
                
                
                # add the short description as an attribute
                if icd_code not in self.graph:
                    self.graph.add_node(icd_code, desc=sdesc)
                # and put the short, the long (if different from the short)
                # and the others descriptions in the indices
                self.definition_index[sdesc] = icd_code
                if icd_code not in self.index_definition:
                    self.index_definition[icd_code] = [sdesc]
                elif ldesc != sdesc:
                    self.index_definition[icd_code].append(ldesc)
                else: 
                    self.index_definition[icd_code].append(sdesc)
        
        # create edges for the graph
        for code in icd_codes:
            if len(code)>3:
                if code[:-1] in self.graph:
                    self.graph.add_edge(code[:-1], code)
                elif code[:-2] in self.graph:
                    self.graph.add_edge(code[:-2], code)
        
        
    def __contains__(self, code):
        """
        Wrapper for `networkx.Graph.has_node()`
        """
        return code in self.graph
    
    def __getitem__(self, code):
        """
        Utility method to access nodes of ICD10 more easily.
        Allows using strings or ids as indices.
        
        Example:
        ```
        > icd10 = ICD10('path/to/icd10')
        > icd10.load_icd10()
        > icd10['A0109']
        {'desc': 'Typhoid fever with other complications'}
        ```
        """
        
        return self.graph.nodes[code]
    
    def predecessors(self, code):
        """
        Wrapper of networkx.digraph.predecessors()
        """
        return list(self.graph.predecessors(code))
    
    def successors(self, code):
        """
        Wrapper of networkx.digraph.successors()
        """
        return list(self.graph.successors(code))
    
    def distance(self, source, target):
        """
        Computes the distance between two nodes.
        """
        
        if nx.has_path(self.graph,source=source, target=target):        
            return nx.shortest_path_length(self.graph,source=source,target=target)
        else:
            return nx.shortest_path_length(self.graph,source=target,target=source)
        
    def is_ancestor(self, source, target):
        """
        Returns True if `source` is an ancestor of `target` in the SNOMED taxonomy.
        """
        
        return nx.has_path(self.graph,source=source, target=target)
    
    def safe_distance(self, source, target):
        """
        Computes the distance between two nodes. If there's not path between the source
        and target node, returns -1.
        """
        
        if nx.has_path(self.graph,source=source, target=target):        
            return nx.shortest_path_length(self.graph,source=source,target=target)
        elif nx.has_path(self.graph,source=target, target=source):  
            return nx.shortest_path_length(self.graph,source=target,target=source)
        else:
            return -1