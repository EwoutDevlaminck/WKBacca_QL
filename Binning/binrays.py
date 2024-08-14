"""THIS FILE CAN BE USED TO CALL THE CYTHON BINNING CODE DIRECTLY FROM TERMINAL.
ALL RELEVANT PARAMETERS CAN BE CHOSEN BELOW. THE DATA IS LOADED AND SAVED FROM
AND TO THE FILES CHOSEN AMONG THE PARAMETERS.
"""

############################################################################
# IMPORT
############################################################################

# Load local modules
from CommonModules.input_data import InputData 
from Binning.modules.binning_interface import binning_pyinterface


############################################################################
# LOAD INPUT PARAMETER FILE INDICATED IN ARGUMENTS AND RUN THE 
# BINNING ROUTINE (IF FILE IS CALLED DIRECTLY)
############################################################################
def call_binning(input_file):

    """ Driver for the binning procedures. """

    # Load input data
    input_data = InputData(input_file)

    # if there are more then one inputfiles in input_data, 
    # split the problem into the single files:
    try:
        if len(input_data.outputfilename) > 0:
            outputfilenames_given = True
        else:
            outputfilenames_given = False
    except:
        outputfilenames_given = False

    
    if outputfilenames_given == True:
        if len(input_data.inputfilename) != len(input_data.outputfilename):
            print('ERROR: IN INPUT DATA FILE, THERE ARE NOT AS MANY INPUTFILES GIVEN AS SUGGESTIONS FOR THE OUTPUTFILENAME.')
            raise

    # copy the original input / outputfilenames
    inputfilenames = input_data.inputfilename
    if outputfilenames_given == True:
        outputfilenames = input_data.outputfilename

    for i in range(0,len(inputfilenames)):
        input_data.inputfilename = inputfilenames[i]
        if outputfilenames_given == True:
            input_data.outputfilename = outputfilenames[i]
                
        binning_pyinterface(input_data)

    # return
    pass
#
# END OF MAIN FUNCTION

############################################################################
# STAND-ALONE RUN
############################################################################
if __name__=='__main__':
    import sys
    input_file = sys.argv[1]
    call_binning(input_file)
#
# END OF FILE
