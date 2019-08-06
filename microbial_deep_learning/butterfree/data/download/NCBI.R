library("curatedMetagenomicData")
source("download_utils")


getNCBINumbers = function(name, dryrun = FALSE) {
    experiments = curatedMetagenomicData(paste0(name,'*'), dryrun = dryrun)
    indices = 1:length(experiments)
    IDs = sapply(indices, function(v) experiments[[v]]$NCBI_accession)
    expNames = sapply(indices, function(v) names(experiments[v]))

    return(list(IDs, expNames))
}

writeNCBINumbers = function(name, fileConnection) {
    experiments = curatedMetagenomicData(paste0(name,'*'), dryrun = FALSE)
    for (i in 1:length(experiments)) {
        write(paste0("Name: ", names(experiments[i])), file = fileConnection, append=TRUE)
        write(experiments[[i]]$NCBI_accession, file = fileConnection, append=TRUE)
    }
}

writeAllNCBINumbers = function(savefile) {
    unique = getUniqueExperiments()
    fileConnection = file(savefile, 'a')
    for (i in 1:3) {
        writeNCBINumbers(unique[i], savefile)
        }
    close(fileConnection)
}
