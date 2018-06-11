#-imports the newest version of the test_data file from
# the dropbox (should have been put there by the phone)
#-runs the feature based model on it, producing an 
# an output graph
#-updates the graph_data file on the dropbox which should
# be shown on the phone app.

import updown
import datacombine_sc
#import DO_NOT_USE_datacombine_aw_changed
import loadtest
import time


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
            # push the new results to the dropbox to be 
            # displayed on the phone
            updown.main('graph_test', './server_local_graph', "push")
            print("\nnew graph files uploaded\n")  
            t1 = time.time()
            print("Total time: {}".format(t1-t0))     
            print("done")
