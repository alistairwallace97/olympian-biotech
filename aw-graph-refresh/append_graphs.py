import datetime

#Append new graphs to old graphs
def append():
    f = open('./server_local_graph/graph_test.txt')
    if f.mode == 'r':
        contents = f.read()
    else:
        print("Error: append_graphs.py could not open \
        graph_test.txt")
    f.close()
    t = datetime.datetime.now()
    f = open('./server_local_graph/local_all_graphs/graph_all_test.txt', "a+")
    f.write("\n\nTime: {}\n\n".format(t))
    f.write(contents)
    f.close 

if __name__ == '__main__':
    append()