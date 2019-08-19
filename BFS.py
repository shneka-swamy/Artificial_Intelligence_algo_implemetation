# This is the basic implementation of BFS and DFS
from collections import defaultdict

def create_edge(dic, start, end):
	dic[start].append(end)

def BFS(dic, start):
	stack = [start]
	visited_list = [False]*len(dic)

	visited_list[start] = True

	print(stack)

	while len(stack) != 0:
		
		for node in dic[stack[0]]:
			if visited_list[node] == False:
				visited_list[node] = True
				stack.append(node)
		stack = stack[1:]

		print(stack)


def main():
	dic = defaultdict(list)

	# This statement is used to form a directed graph
	# The first argument is the dictionary, the source and the destination vertex
	#create_edge(dic, 0, 1)
	#create_edge(dic, 0, 2)
	#create_edge(dic, 2, 0)
	#create_edge(dic, 1, 2)
	#create_edge(dic, 2, 3)
	#create_edge(dic, 3, 3)

	# This is the attempt to use undirected graph
	create_edge(dic, 0, 1)
	create_edge(dic, 0, 2)
	create_edge(dic, 1, 3)
	create_edge(dic, 1, 4)
	create_edge(dic, 2, 4)
	create_edge(dic, 3, 4)
	create_edge(dic, 3, 5)
	create_edge(dic, 4, 5)

	create_edge(dic, 1, 0)
	create_edge(dic, 2, 0)
	create_edge(dic, 3, 1)
	create_edge(dic, 4, 1)
	create_edge(dic, 4, 2)
	create_edge(dic, 4, 3)
	create_edge(dic, 5, 3)
	create_edge(dic, 5, 4)

	print(dic)

	print("Running BFS:")

	BFS(dic, 0)

if __name__ == "__main__":
	main()
