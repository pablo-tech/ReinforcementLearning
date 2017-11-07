### SMALL DATASET: model-based learning

using RDatasets, DataFrames
using CSV

### INPUT: dataset

function getDataset(input_filename::String)
    dataset = readtable(input_filename, header=true)
    return dataset
end

function getNumInputRows(input_dataset::DataFrames.DataFrame)
    columnNames = names(input_dataset)
    aColumn = columnNames[1]
    numInputRows = length(input_dataset[aColumn])
end

function input(input_filename::String)
    input_dataset = getDataset(input_filename)
    # input_dataset = dataset("datasets",input_filename)

    columnNames = names(input_dataset)

    for rowNum in 1:getNumInputRows(input_dataset)  # first row is header
        row = input_dataset[rowNum:rowNum,:]  # just the one row, every column
        for columnName in columnNames
            ## PROCESS ROW: reinforced learning
            # print("(", columnName, ",", row, ")")
        end
    end
end


### OUTPUT: policy
function output(input_filename::String)
    output_filename = input_filename * ".policy"

    try
        rm(output_filename)
    except
        println("could not delete inexistent file: " * input_filename)
    end

    out_file = open(output_filename, "w")

    input_dataset = getDataset(input_filename)
    columnNames = names(input_dataset)
    #aColumn = columnNames[1]
    #numOutputRows = length(input_dataset[aColumn])  # TODO

    for rowNum in 1:getNumInputRows(input_dataset)
        row = input_dataset[rowNum:rowNum,:]  # just the one row, every column
        ## PROCESS: output policy, no header row
        println(out_file, rowNum)
        #println(out_file, row)
    end

    close(out_file)
end


### CHECK: policy vs dataset
# check policy file length == number of states
function check(output_filename::String, expected_length::Int64)
    output_readback = getDataset(output_filename)
    numPolicyRows = getNumInputRows(output_readback)

    @printf("\n**** EOL policy rows vs expected **** %s: \t %d vs\t %d \n", output_filename, numPolicyRows, expected_length)
end


