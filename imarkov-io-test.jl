### DATASET: TEST


### Dataframe
using RDatasets, DataFrames

columnNames = names(anscombe)

numColumns = length(anscombe)
numRows = length(anscombe[:Y2])

# indexes from 1
columnNames[1]

# head() or tail() to see parte of the dataset

# showcols() type of each column


### File I/O
# https://cbrownley.wordpress.com/2015/05/29/reading_writing_csv_with_r_python_julia/

using CSV

input_filename = "small.csv"

# INPUT
for line in eachline(input_filename)
           print(line)
       end

# OUTPUT
# write(output_filename, line)



