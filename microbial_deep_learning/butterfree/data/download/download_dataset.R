# This scripts downloads files from curatedMetagenomicData to CSV format

args <- commandArgs()
saveDir <- args[6]
print(paste("Downloading files from curatedMetagenomicData to ", saveDir))
source("download_utils.R")

unique <- getUniqueExperiments()

for (i in 1:length(unique)) {
  tryCatch({
    writeNamedExperiment(unique[i], saveDir)
    print(paste("Wrote CSVs for ", unique[i]))
  },
  error = function(e) {print(paste("Failed to write CSVs for ", unique[i]))}
)
}
print(paste("Wrote CSVs from curatedMetagenomicData to ", saveDir))
