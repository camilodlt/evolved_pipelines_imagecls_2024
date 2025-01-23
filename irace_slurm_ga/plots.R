setwd("~/.julia/dev/PCAM/")
library(plotly)
library(irace)
library(iraceplot)
library(ggridges)
library(ggplot)

iraceResults <- irace::read_logfile("irace_slurm_ga_noracing/irace.Rdata")
######## PARALLEL COORDS CONFIGS ##########

t <- list(
  family = "Computer Modern",
  size = 22,
  color = 'black')

g = parallel_coord(iraceResults,
                   iterations = 1:iraceResults$state$nbIterations,
                   only_elite = FALSE, color_by_instances = FALSE,
                   param_names = c("mutation_rate","n_nodes","n_elite", "n_new",
                                   "toursize", "n_samples", "n_repetitions","acc_w"))

g$x$attrs[[2]]$dimensions[[1]]$label = "ID"
g$x$attrs[[2]]$dimensions[[2]]$label = "Mutation Rate"
g$x$attrs[[2]]$dimensions[[3]]$label = "Nodes"
g$x$attrs[[2]]$dimensions[[4]]$label = "# Elite"
g$x$attrs[[2]]$dimensions[[5]]$label = "# Offspring"
g$x$attrs[[2]]$dimensions[[6]]$label = "Tournament size"
g$x$attrs[[2]]$dimensions[[7]]$label = "Samples/Batch"
g$x$attrs[[2]]$dimensions[[8]]$label = "Sampling Repetitions"
g$x$attrs[[2]]$dimensions[[9]]$label = "ρ"

g$x$attrs[[2]]$line$colorscale = "Portland"
#Blackbody,Bluered,Blues,Cividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.
g <- g %>% layout(font = t)

g$x$attrs[[2]]$dimensions[[1]]$visible = FALSE
g

if (!require("processx")) install.packages("processx")
#kaleido(g, "irace_slurm_ga_noracing/all_configs.svg")
save_image(g, file = "irace_slurm_ga_noracing/all_configs.png", dpi = 300, dpi = 300,width=1000, height=600, scale =1)
save_image(g, file = "irace_slurm_ga_noracing/all_configs.svg")

######## BOXPLOTS ##########

g <- boxplot_performance(iraceResults$experiments,  rpd = FALSE, boxplot = TRUE)
c <- g %>% plotly::layout(paper_bgcolor='#FFFFFF', plot_bgcolor = "#FFFFFF", font = t)
c$x$layout$xaxis$linecolor = "#000000"
c$x$layout$xaxis$linewidth = 1
c$x$layout$xaxis$showline = TRUE

c$x$layout$yaxis$linecolor = "#000000"
c$x$layout$yaxis$linewidth = 1
c$x$layout$yaxis$showline = TRUE

c$x$layout$annotations[[3]] = NULL
c$x$layout$annotations[[2]]$text = "Negative Validation Accuracy (minimized)"
c$x$layout$shapes[[2]] = NULL # remove the banner (facet grid)


c$x$data[[1]]$line$color[1] = "rgba(44, 45, 97, 1)"
c$x$data[[2]]$line$color[1] = "rgba(44, 45, 97, 1)"
c$x$data[[3]]$marker$color[1] = "rgba(153,153,153,1)"
c$x$data[[3]]$marker$line$color[1] = "rgba(153,153,153,1)"
c$x$data[[3]]$marker$opacity = 0.3

c$x$data[[2]]$marker$outliercolor[1] = "rgba(153,153,153,1)"
c$x$data[[2]]$marker$opacity = 0.3
#c$x$data[[2]]$marker$line$width = 20

#c$x$data[[2]]$marker$line$outlierwidth = 0. # no border for the outlier
c$x$data[[2]]$marker$line$color[1] = "rgba(153,153,153,1)"
c$x$data[[2]]$marker$line$outliercolor[1] = "rgba(153,153,153,1)"

c$x$data[[3]]$marker$opacity[1] = 0.3
c$x$data[[4]]$marker$opacity[1] = 0.3

c$x$layout$yaxis$range = c(-0.8, -0.4865906)
c$x$layout$yaxis$ticktext = c("-0.75", "-0.7", "-0.6", "-0.5")
c$x$layout$yaxis$tickvals = c( -0.75, -0.7, -0.6, -0.5)

save_image(c, file = "irace_slurm_ga_noracing/val_metric_all_configs.png", dpi = 300,width=1000, height=600, scale =1)
save_image(c, file = "irace_slurm_ga_noracing/val_metric_all_configs.svg",width=1000, height=600, scale =1)

c######## WINNER CONFIGURATIONS ##########
# SEE WINNERS PER ITERATION
iraceResults$iterationElites

# Elite Configurations Irace # winner 68
getConfigurationById(iraceResults, 68)
# Mean score 
round(mean(iraceResults$experiments[,68]),3)
round(sd(iraceResults$experiments[,68]), 2)

# Config 30 was also a winner during iteration 5
getConfigurationById(iraceResults, 30)
# Mean score 
round(mean(iraceResults$experiments[,30]),3)
round(sd(iraceResults$experiments[,30]), 2)

# Config 1 was also a winner during all others iterations
getConfigurationById(iraceResults, 1)
# Mean score 
round(mean(iraceResults$experiments[,1]),3)
round(sd(iraceResults$experiments[,1]), 2)

# Other good configs where finalists (11, 18)


# 

