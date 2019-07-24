import json
from collections import deque

class Node(object):
    def __init__(self,name : str,_id,num_id : int):
        self.name = name
        self._id = _id
        self.num_id = num_id
        self.childs = []
        self.parents = []
        self.childs_dict = {}
        self.parents_dict = {}
        self.num_childs = 0
        self.num_parents = 0
        
    def json(self):
        json = {
            "name":self.name,
            "_id":self._id,
            "num_id":self.num_id,
            "childs":self.childs,
            "childs_dict":self.childs_dict,
            "parents":self.parents,
            "parents_dict":self.parents_dict
        }
        return json
    
    def add_child(self, v : int, name : str):
        self.childs.append(v)
        self.childs_dict[name] = self.num_childs
        self.num_childs+=1
    def add_parent(self, v : int, name : str):
        self.parents.append(v)
        self.parents_dict[name] = self.num_parents
        self.num_parents+=1
    @staticmethod
    def from_json(json : dict):
        res = Node(json["name"],json["_id"],json["num_id"])
        res.childs = json["childs"]
        res.childs_dict = json["childs_dict"]
        res.num_childs = len(res.childs)
        res.parents = json["parents"]
        res.parents_dict = json["parents_dict"]
        res.num_parents = len(res.parents)
        return res
    
class TopicsGraph(object):
    def __init__(self):
        self.nodes = []
        self.v = 0
        self.e = 0
        self.nodes_name_dict = {}
        self.nodes_id_dict = {}
    def add_node(self,name : str, _id : str):
        self.nodes.append(Node(name,_id,self.v))
        self.nodes_name_dict[name] = self.v
        self.nodes_id_dict[_id] = self.v
        self.v+=1
    def add_edge(self,u : int, v : int):
        self.nodes[u].add_child(v,self.nodes[v].name)
        self.nodes[v].add_parent(u,self.nodes[u].name)
        self.e+=1
    def save(self,graphFile : str):
        data = {"nodes_name_dict":self.nodes_name_dict,"nodes_id_dict":self.nodes_id_dict,"nodes":None}
        nodes = []
        for node in self.nodes:
            nodes.append(node.json())

        file = open(graphFile,"w")
        data["nodes"] = nodes
        json.dump(data,file)
        file.close()
        
    def restore(self,graphFile : str):
        with open(graphFile,"r",encoding="utf8") as gfile:
            data = json.loads(gfile.read())
            nodes = data["nodes"]
            
            self.nodes_name_dict = data["nodes_name_dict"]
            self.nodes_id_dict = data["nodes_id_dict"]
            
        for item in nodes:
            self.nodes.append(Node.from_json(item))
            self.v += 1
            self.e += len(item["childs"])
            
    def expand_tags(self,tags: list):
        visited = [False]*self.v
        for tag in tags:
            name = tag
            if not name in self.nodes_name_dict:
                continue
            
            index = self.nodes_name_dict[name]
            while len(self.nodes[index].parents) == 1 and not visited[index]:
                visited[index] = True
                index = self.nodes[index].parents[0]
                
            visited[index] = True
        result = [self.nodes[i].name for i in range(self.v) if visited[i]]
        return result
    def expand_tags_down_in_hierarchy(self,tags : list):
        visited = [False]*self.v
        q = deque(tags)
        s = set(tags)
        for tag in tags:
            visited[self.nodes_name_dict[tag]] = True
        
        
        while len(q)>0:
            name = q.popleft()
            
            if not name in self.nodes_name_dict:
                continue
            
            index = self.nodes_name_dict[name]
            
            flag = False
            for child in self.nodes[index].childs:
                if self.nodes[child].name in s:
                    flag = True
                    break
            if flag:
                continue
            
            
            for child in self.nodes[index].childs:
                if not visited[child]:
                    visited[child] = True
                    q.append(self.nodes[child].name)
                
        result = [self.nodes[i].name for i in range(self.v) if visited[i]]
        return result
        
        
        
        pass
    def specific_print(self,outfile):
        file = open(outfile,"w",encoding="utf8")
        file.write("{}\n".format(self.v))
        childs = []
        for node in self.nodes:
            childs.append(node.childs)
            file.write("{}\n{}\n{}\n".format(node.name,node._id,node.num_id))
        for c in childs:
            for node in c:
                file.write("{} ".format(node))
            file.write("\n")
        file.close()
    def updateGraph(self,topicsFile):
        file = open(topicsFile,"r",encoding="utf8")
        data = []
        for line in file:
            d = json.loads(line)
            data.append(d)
            
        file.close()
        dict_id = {}
        dict_name = {}
        int_id = self.v
        
        for item in data:
            if not item["_id"]["$oid"] in dict_id:
                dict_id[item["_id"]["$oid"]] = int_id
                dict_name[int_id]  = item["title"]
                int_id+=1
                self.add_node(name = item["title"],_id = item["_id"]["$oid"])
                
        for item in data:
            v = dict_id[item["_id"]["$oid"]]
            flag = False
            for parent in item["topicParents"]:
               if parent["$oid"] in dict_id: 
                   u = dict_id[parent["$oid"]]
                   self.add_edge(u,v)
                   flag = True
               elif parent["$oid"] in self.nodes_id_dict:
                   u = self.nodes_id_dict[parent["$oid"]]
                   self.add_edge(u,v)
                   flag = True
                   
            if len(item["topicParents"]) == 0 or not flag: 
               self.add_edge(0,v)
               
        pass
    @staticmethod
    def parseJsonAndCreateGraph(topicsFile):
        file = open(topicsFile,"r",encoding="utf8")
        data = []
        for line in file:
            d = json.loads(line)
            data.append(d)
            
        file.close()
        dict_id = {}
        dict_name = {}
        int_id = 1
        g = TopicsGraph()
        g.add_node(name="root",_id="root")
        for item in data:
            if not item["_id"]["$oid"] in dict_id:
                dict_id[item["_id"]["$oid"]] = int_id
                dict_name[int_id]  = item["title"]
                int_id+=1
                g.add_node(name = item["title"],_id = item["_id"]["$oid"])
        for item in data:
            v = dict_id[item["_id"]["$oid"]]
            flag = False
            for parent in item["topicParents"]:
               if parent["$oid"] in dict_id: 
                   u = dict_id[parent["$oid"]]
                   g.add_edge(u,v)
                   flag = True
            if len(item["topicParents"]) == 0 or not flag: 
               g.add_edge(0,v)
               
        
        return g
    def get_list_nodes(self):
        return [node.name for node in self.nodes if node.name != "root"]
    
    
    
        
            
            
            
            
            