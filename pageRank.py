import pandas as pd
import numpy as np
import networkx as nx
import argparse
import matplotlib.pyplot as plt
import time
from scipy.sparse import csr_matrix as csr



class Item:

   def __init__(self, name, rank):
      self.name = name
      self.rank = rank

   def __lt__(self, other):
      return self.rank > other.rank

   def __str__(self):
      return str(self.name) +  " with PageRank: " + str(self.rank)

def parse_small(file_path):
   f = open(file_path, "r")
   G = nx.DiGraph()
   for x in f.readlines():
      line = x.split(",")
      G.add_edge(line[0], line[2])
   f.close()
   return G

def parse_snap(file_path):
   G = nx.DiGraph()
   f = open(file_path, "r")
   for x in f.readlines():
      line = x.split()
      if line[0] != '#':
         G.add_edge(line[0], line[1])
   f.close()
   return G

def get_df(file_path, d):
   f = open(file_path, "r")
   lines = f.readlines()
   num_nodes = int(lines[2].split()[2])
   rows = []
   columns = []
   data = []
   for x in lines[4:]:
      line = x.split()
      rows.append(int(line[0]))
      columns.append(int(line[1]))
      data.append(1)
   matrix = csr((data, (rows, columns)), shape=(num_nodes, num_nodes))
   sums = np.sum(matrix, axis=1)
   sums[sums == 0] = 1
   matrix = matrix.multiply(d/sums)
   f.close()
   return matrix, num_nodes

def create_df(file_path, d, s):
   if s == None:
      G = parse_small(file_path)
   else:
      G = parse_snap(file_path) 
   df = nx.to_pandas_adjacency(G, sorted(G.nodes))
   for x in range(df.shape[0]):
      start = time.time()
      the_sum = df.iloc[x].sum()
      if the_sum > 0:
         df.iloc[x] = d * (df.iloc[x] / the_sum)
   df.loc[-1] = [1/df.shape[0]] * df.shape[0]
   return df.transpose()

def initialize_pk(df, d):
   pk = [1/df.shape[0] for x in range(df.shape[0])]
   pk.append(1-d)
   return np.array(pk)

def calculate_new_pk(df, pk, d):
   new_pk = df.dot(pk)
   new_pk = new_pk.append(pd.Series([1-d]))
   return np.array(new_pk)

def check_stop(pk, new_pk, epsilon):
   val = ((pk - new_pk) ** 2).sum()
   return val < epsilon

def print_page_ranks(lst, new_pk, read_end, rank_end, iterations, ep):
   print("Time to process read in data and build graph:", read_end)
   print("Time to compute PageRank for each node in graph:", rank_end)
   print("Number of iterations until convergence:", iterations + 1)
   print("Threshold value:", ep) 
   print("\nPageRanks:")
   items = [Item(lst[x], new_pk[x]) for x in range(len(lst))]
   items.sort()
   for x in items:
      print(x)

def main():
   parser = argparse.ArgumentParser(description="Run Page rank")
   parser.add_argument("csv_path")
   parser.add_argument("-s", "--snap", required=False)
   parser.add_argument("-e", "--ep", required=True)
   args = parser.parse_args()
   file_path = args.csv_path
   d = .85
   read_start = time.time()
   iterations = 0
   if args.snap == None or int(args.snap) == 0:
      df = create_df(file_path, d, args.snap)
      read_end = time.time() - read_start
      pk = initialize_pk(df, d)
      new_pk = calculate_new_pk(df, pk, d)
      iterations = 0
      rank_start = time.time()
      while not check_stop(pk, new_pk, float(args.ep)):
         pk = new_pk
         new_pk = calculate_new_pk(df, pk, d)
         iterations += 1
      rank_end = time.time() - rank_start
      print_page_ranks(df.index, new_pk, read_end, rank_end, iterations, float(args.ep))
   else:
      matrix, num_nodes = get_df(file_path, d)
      #print(matrix.sum(axis=0))
      #print(matrix.sum(axis=1))
      read_end = time.time() - read_start
      const = (1-d) * 1/num_nodes
      init_pk = np.array([1/num_nodes for x in range(num_nodes)])
      new_pk = matrix.dot(init_pk) + const
      rank_start = time.time()
      while not check_stop(init_pk, new_pk, float(args.ep)):
         init_pk = new_pk
         new_pk = matrix.dot(init_pk) + const
         iterations += 1
      rank_end = time.time() - rank_start
      print_page_ranks([x for x in range(num_nodes)], new_pk, read_end, rank_end, iterations, float(args.ep))


if __name__ == '__main__':
   main()
