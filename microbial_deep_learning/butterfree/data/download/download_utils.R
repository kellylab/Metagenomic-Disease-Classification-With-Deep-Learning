# Writes an object's data parameters to file, along with the parameters associated with any attributes of the object
# It does so by creating a directory tree and recursively calling itself on the attributes of attributes of any object

# """ FOR AN EXPRESSIONSET OBJECT ACCESS THE $ ATTIRUBTES USING varLabels(). THESE ARE CALLED ANNOTATED DATA FRAME INSTANCES ASSOCIATED TO THE OBJECT"""

library("curatedMetagenomicData")
library("stringr")

eh <- ExperimentHub()

getExperimentNames = function() {
    names = ls("package:curatedMetagenomicData")
    return(names)
}

getUniqueExperiments = function() {
    # Returns the overall experiment name for each dataset in list.
    names = getExperimentNames()
    removeTags = str_extract_all(names, ".+(?=[.].+[.])")
    uniqueNames = unique(removeTags)
    return(uniqueNames)
}

writeNamedExperiment = function(name, saveDirectory, dryrun = FALSE) {
    # Writes all experimens under a certain name to file
    # To avoid headaches, call getUniqueExperiments() to get a list of unique names
    # And then use one of those names as the argument for this function.
    experiments = curatedMetagenomicData(paste0(name,"*"), dryrun = dryrun) # Look for all files that start with name
    # Set up directories
    currentDirectory = saveDirectory
    writeDirectory = paste(currentDirectory, name, sep='/')
    dir.create(writeDirectory)
    # Write phenotype data and other metadata
    experiment = experiments[[1]]
    phenoData = experiment@phenoData
    filename = paste(writeDirectory, 'phenoData', sep='/')
    writeObject(phenoData, filename)

    # Write assay data for each experiment
    for (i in 1:length(experiments)) {
        # All of the files are of the type author_name.assay_type.body_site
        # name is the author_name. We need to extract assay_type adn body_site
        experiment = experiments[[i]]
        name = names(experiments[i])
        splits = unlist(strsplit(name, '[.]'))
        authorName = splits[1]
        assayName = splits[2]
        bodySite = splits[3]
        writeDirectory = paste(currentDirectory, authorName, bodySite, sep='/')
        dir.create(writeDirectory)
        filename = paste(writeDirectory, assayName, sep='/')
        print(paste0('Writing: ', filename))
        writeObject(experiment@assayData, filename)
    }
}

writeObjectsInDirectory = function(eh, start = 1, directory) {
    workingDirectory <- directory
    for (i in start:length(eh)) {
        filename = paste(workingDirectory,names(eh[i]),sep='/')
        writeObject(eh[i], names(eh[i]))
    }
}

tryWriteObjectsInDirectory = function(eh, start = 1, directory) {
  workingDirectory <- directory
  for (i in start:length(eh)) {
    filename <- paste(workingDirectory,names(eh[i]),sep='/')
    tryCatch({
      object <- eh[[i]]
      writeObject(eh[[i]], filename)
      rm(object)
    },
    error = function(e) {print(paste0("Failed to load and write EH", i))}
  )
  }
}

writeExpressionSet = function(eset, filename) {
  attributeKeys <- varLabels(eset)
  slotKeys <- slotNames(eset)
  writeKeys(eset, filename, attributeKeys, slotKeys)
  }

writeExperimentHub = function(ehub, filename) {
    attributeKeys <- c('ah_id', 'title', 'dataprovider', 'species', 'taxonomyid', 'genome', 'description', 'coordinate_1_based',
      'maintainer', 'rdatadateadded', 'preparerclass', 'tags', 'rdataclass', 'sourceurl', 'sourcetype')
    #slotKeys = strsplit(slotNames(object), " +") # Get the slot names in a list
    slotKeys <- c("hub", "cache", ".db_path", ".db_index", ".db_uid")
    writeKeys(ehub, filename, attributeKeys, slotKeys)
  }

writeKeys = function(object, filename, attributeKeys, slotKeys) {
    print(object)
    newDirectory <- paste(filename, sep="/")
    dir.create(newDirectory) # Will do nothing if directory already exists
    print(paste("Working on",newDirectory,sep=" "))

  for (key in attributeKeys) { # Loop over $ attributes and write to file
      tryCatch({
        childObject <- eval(parse(text=paste0("object$",key)))  # This replaces key in object$key with the name of the key
        print(paste0("object$",key))
        newFilename <- paste(filename, key, sep="/")
        writeObject(childObject, newFilename)
      },
        error = function(e) {print(paste0('Failed to save ', filename, ', ', key))} )
  }

    for (key in slotKeys) { # Loop over @ attributes and write to file by expanding tree structure
      tryCatch({
        childObject <- slot(object, key)
        newFilename <- paste(filename, key, sep="/")
        writeObject(childObject, newFilename)
      },
      error = function(e) {print(paste0('Failed to save ', filename, ', ', key))})
    }
}

writeData = function(object, filename) { # Writes lists, dataframes, etc. to csv
    print(paste0("printing: ", filename))
    tryCatch(write.csv(object, filename), error = function(e){})
}

writeObject = function(object, filename) { # Checks type of object and determines how to write it
    if (typeof(object) == 'character') {
        write.csv(object, filename)
    }
    else if (typeof(object) == 'integer') {
        write.csv(object, filename)
    }
    else if (class(object) == 'AnnotatedDataFrame') {
        print('ok')
        write.csv(as(object, 'data.frame'), filename)
    }
    else if (class(object) == 'environment') {
        write.csv(object$exprs, filename)
    }
    else if (class(object) == 'ExpressionSet') {
        writeExpressionSet(object, filename)
    } else if (class(object) == 'ExperimentHub') {
        writeExperimentHub(object, filename)
    } else {} # Ignore whatever doesn't fit.
}

hasChildren = function(object) { # Checks if a given object has $ or @ attributes which must be further processed
    attributeKeys <- c()
    slots = length(attributes(object))
    if (isEmpty(attributeKeys) & slots == 0) {
        bool <- FALSE
    } else {
        bool <- TRUE
    }

    return(bool)
}

isEmpty = function(list) {
    bool <- (length(list) == 0)

    return(bool)
}

getIndices = function(dataFrame, datasetName) { # Finds indices in dataFrame that have name datasetName
    bool <- (dataFrame$dataset_name == datasetName)
    indices <- which(bool)

    return(indices)
    }

findEverySlotEver = function(eh) { # Finds all slots of an object and returns as a list.
  lollipop <- c()
  for (i in 1:length(eh)) {
    a <- eh[[i]]
    keys <- slotNames(a)
    for (key in keys) {
      if (! (class(a) %in% lollipop)) {
        lollipop <- append(lollipop, class(a))
      }
      print(lollipop)
    }
    rm(a)
  }
  return(lollipop)
}
