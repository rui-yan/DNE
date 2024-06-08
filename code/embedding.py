from models.dne import DNE
from models.svd import SVD
from models.node2vec import Node2Vec
from models.grarep import GraRep
from models.line import LINE
from models.hope import HOPE
from models.netmf import NetMF
from models.le import LE
from models.lle import LLE
from utils.walker import RandomWalker


class NodeEmbedding:
	def __init__(self, args, graph, name):
		self.args = args
		self.graph = graph
		self.name = name
		
		self._simulate_walks(
			n=args.walk_number, 
			l=args.walk_length, 
			p=args.p, 
			q=args.q, 
			use_rejection_sampling=True
			)
	
	def _simulate_walks(self, n, l, p, q, use_rejection_sampling):
		walker = RandomWalker(self.graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)
		walker.preprocess_transition_probs()
		self.walks = walker.simulate_walks(num_walks=n, walk_length=l, workers=1, verbose=1)
	
	def get_embeddings(self, method, node_attributes=None, embed_size=128):
		
		print(f'Generate {method} embeddings for {self.name}')
		if method == 'SVD':
			model = SVD(self.graph, embed_size=embed_size)
			embeddings = model.get_embeddings()
		
		elif method == 'DNE':
			print('node_attributes: ', node_attributes)
			if node_attributes is None:
				model = DNE(self.graph, hidden_dim=embed_size, pos_embed_size=256, 
                       pos_embed=self.args.pos_embed, node_attributes=None, walks=self.walks)
			else:
				model = DNE(self.graph, hidden_dim=embed_size, pos_embed_size=256, 
                   		feat_size=node_attributes.shape[1], 
                  		pos_embed=self.args.pos_embed, 
                    	node_attributes=node_attributes, 
                    	walks=self.walks)
			model.train(batch_size=3000, epochs=5)
			embeddings = model.get_embeddings()

		elif method == 'N2V':
			model = Node2Vec(self.graph, embed_size, self.walks)
			model.train(window_size=5, iter=3)
			embeddings = model.get_embeddings()
		
		elif method == 'GraRep':
			model = GraRep(self.graph, embed_size)
			model.train(order=4)
			embeddings = model.get_embeddings()
		
		elif method == 'HOPE':
			model = HOPE(self.graph, embed_size)
			model.fit(self.graph)
			embeddings = model.get_embeddings()

		elif method == 'LINE':
			model = LINE(self.graph, embed_size, negative_ratio=5, order='second')
			model.train(batch_size=3000, epochs=50, verbose=0)
			embeddings = model.get_embeddings()
	
		elif method == 'NetMF':
			model = NetMF(self.graph, embed_size, window_size=5)
			embeddings = model.get_embeddings()

		elif method == 'LE':
			model = LE(self.graph, embed_size)
			embeddings = model.get_embeddings()
	
		elif method == 'LLE':
			model = LLE(self.graph, embed_size)
			embeddings = model.get_embeddings()
		
		self.embeddings = embeddings
		
		return self.embeddings
	
	def save_embeddings(self, filename):
		fout = open(filename, 'w')
		node_num = len(self.embeddings.keys())
		fout.write("{} {}\n".format(node_num, self.embed_size))
		for node, vec in self.embeddings.items():
			fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
		fout.close()