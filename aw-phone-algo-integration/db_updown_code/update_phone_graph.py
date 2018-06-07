# test model
#-imports the newest version of the test_data file from
# the dropbox (should have been put there by the phone)
#-runs the feature based model on it, producing an 
# an output graph
#-updates the graph_data file on the dropbox which should
# be shown on the phone app.

import updown
import datacombine
#import loadtest


if __name__ == '__main__':
    if(updown.main()):
        print("\n\nThe file had changed so running datacombine\n")
        datacombine.main('./server_local_test_data/', False)
    #print("\n\n\n")
    #loadtest.main()
    else:
        print("\n\nfile has not changed")
    print("done")
