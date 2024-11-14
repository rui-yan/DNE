import sys
sys.path.append(".")

from models.dne import DNE
from models.svd import SVD
from models.node2vec import Node2Vec
from models.grarep import GraRep
# from models.line import LINE
from models.hope import HOPE
from models.netmf import NetMF
from models.le import LE
from models.lle import LLE


class NodeEmbedding:
	def __init__(self, args=None, graph=None, name='Train_graph'):
		self.args = args
		self.graph = graph
		self.name = name
	
	def get_embeddings(self, method, feat=None, embed_size=128):
		print(f'Generate {method} embeddings for {self.name}')
		if method == 'SVD':
			model = SVD(self.graph, embed_size=embed_size)
			embeddings = model.get_embeddings()
		
		elif method == 'DNE':
			batch_size = self.args.batch_size if self.args else 5120
			epochs = self.args.epochs if self.args else 10
			walk_number = self.args.walk_number if self.args else 100
			walk_length = self.args.walk_length if self.args else 10
			p = self.args.p if self.args else 1.0
			q = self.args.q if self.args else 1.0
			
			if feat is None:
				model = DNE(self.graph, hidden_dim=embed_size)
			else:
				model = DNE(self.graph, feat=feat, hidden_dim=embed_size)
			
			model.train(batch_size=batch_size, 
               			epochs=epochs,
               			walk_number=walk_number, 
                  		walk_length=walk_length, 
                    	p=p, 
                     	q=q)
			
			embeddings = model.get_embeddings()

		elif method == 'N2V':
			model = Node2Vec(self.graph, embed_size)
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

		# elif method == 'LINE':
		# 	model = LINE(self.graph, embed_size, negative_ratio=5, order='second')
		# 	model.train(batch_size=3000, epochs=50, verbose=0)
		# 	embeddings = model.get_embeddings()
		
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