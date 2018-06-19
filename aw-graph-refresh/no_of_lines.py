f = open('./server_local_test_data/test_data.txt', 'r')
content = f.readlines()
counter = 0
for line in content:
    counter += 1

print("there are ", counter, " lines")