# function for appending a load of lists within a lists


def ls_of_ls_appender(lol):
    out_lst = [0]
    for i in range(len(lol)):
        for j in range(len(lol[i])):
            out_lst.append(lol[i][j])
    return out_lst[1:]

lol_test = [[1,2,3],[4,5,6],[7,8,9],[-1,-2,-3]]
test = ls_of_ls_appender(lol_test)

