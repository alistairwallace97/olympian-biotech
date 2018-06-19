#-imports the newest version of the test_data file from
# the dropbox (should have been put there by the phone)
#-runs the feature based model on it, producing an 
# an output graph
#-updates the graph_data file on the dropbox which should
# be shown on the phone app.

import updown
import datacombine_sc
import loadtest
import time
import append_graphs as ag


if __name__ == '__main__':
    while(True): #polling the sync input
        # check if any new test data has arrived
        if(updown.main('test_data', './server_local_test_data', "pull")):
            t0 = time.time()
            print("\nThe file has changed so running datacombine\n")
            # if so convert to the right form
            datacombine_sc.main('update_phone_graph')
            print("\ndata combined/converted\n")
            # run the model on it to get the results
            loadtest.main()
            print("\nmodel run\n")
            # pull all graph results from dropbox
            updown.main('graph/all_graphs', './server_local_graph/local_all_graphs', 'pull')
            #append to previous graphs
            ag.append()
            # push the new results to the dropbox to be 
            # displayed on the phone
            updown.main('graph', './server_local_graph', "push")
            # push new all graphs file to db
            updown.main('graph/all_graphs', './server_local_graph/local_all_graphs', 'push')
            print("\nnew graph files uploaded\n")  
            t1 = time.time()
            print("Total time: {}".format(t1-t0))     
            print("done")
