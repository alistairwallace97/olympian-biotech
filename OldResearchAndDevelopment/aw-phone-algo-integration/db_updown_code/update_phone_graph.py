#-imports the newest version of the test_data file from
# the dropbox (should have been put there by the phone)
#-runs the feature based model on it, producing an 
# an output graph
#-updates the graph_data file on the dropbox which should
# be shown on the phone app.

import updown
import datacombine
import loadtest


if __name__ == '__main__':
    # check if any new test data has arrived
    if(updown.main('test_data', './server_local_test_data', "pull")):
        print("\nThe file has changed so running datacombine\n")
        # if so convert to the right form
        datacombine.main('./server_local_test_data/', False)
        print("\ndata combined/converted\n")
        # run the model on it to get the results
        loadtest.main()
        print("\nmodel run\n")
        # push the new results to the dropbox to be 
        # displayed on the phone
        updown.main('graph', './server_local_graph', "push")
        print("\nnew graph files uploaded\n")       
    else:
        print("\n\ntest_data file has not changed")
    print("done")
