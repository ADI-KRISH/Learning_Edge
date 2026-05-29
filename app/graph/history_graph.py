import networkx as nx 
import json 

class Node:
    def __init__(self, topic=None, human_chat=None, ai_chat=None, topic_head=None):
        self.topic = topic
        self.chat_history = {}
        self.i = 0
        self.children = {}

class StartNode:
    def __init__(self):
        self.children = {}

    def add_node(self, node: Node):
        self.children[node.topic] = node 

class ChatHistoryGraph:
    def __init__(self):
        self.root = StartNode()
    
    def add_branch(self, node: Node):
        self.root.add_node(node)
        return self.root
    
    def add_chat_to_node(self, ai_chat, human_chat, node: Node):
        target_node = self.search_branch(node.topic)
        if target_node:
            target_node.chat_history[target_node.i] = (human_chat, ai_chat)
            target_node.i += 1
        else:
            node.chat_history[node.i] = (human_chat, ai_chat)
            node.i += 1
            self.add_branch(node)
    
    def search_branch(self, topic: str):
        def dfs(n):
            if n is None:
                return None
            if hasattr(n, 'topic') and n.topic == topic:
                return n
            for child in getattr(n, 'children', {}).values():
                result = dfs(child)
                if result is not None:
                    return result
            return None
        
        return dfs(self.root)


    
    
        
        


        
