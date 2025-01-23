library(iraceplot)
iraceResults <- irace::read_logfile("irace.Rdata")


# PLOT SETTINGS TRIED
parallel_coord(iraceResults,
  iterations = 1:iraceResults$state$nbIterations,
  color_by_instances = FALSE,
  only_elite = FALSE
)

# Parameter Sampling Freq 
sampling_frequency(iraceResults)
