##########
#
# Show statistics on one CV fold
#
#
#
##########
library(getopt)
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(jsonlite))
#suppressPackageStartupMessages(library(gridExtra))
#library(PRROC)

spec = matrix(c(
'help' ,  'h', 0, "logical",
'input',  'i', 1, "character",
'output', 'o', 1, "character"
), byrow=TRUE, ncol=4)
opt = getopt(spec)

# if help was asked for print a friendly message
# and exit with a non-zero error code
if ( !is.null(opt$help) | is.null(opt$input) | is.null(opt$output)) {
    cat(getopt(spec, usage=TRUE))
    q(status=1)
}

# validation stats
long<-fromJSON(opt$input)$validation


# convert minibatches to fractional epoch
sc<-long%>%summarize(epoch=max(epoch),minibatch=max(ctr))%>%transmute(sc=epoch/minibatch)

long<-long%>%mutate(f_epoch=ctr*sc$sc)


png(opt$output, width=20, height=10, res=200, units = "in", pointsize = 12, type='cairo', antialias = "default")

long_m<-long%>%gather(`acc`,`prec`,`auc`,`tpr`,`tnr`,key='measure', value='score')

ggplot(long_m,aes(y=score, x=f_epoch,color=measure,fill=measure))+
    theme_bw(base_size = 16)+
    geom_smooth()+xlab('Epoch')+
    ggtitle("Performance on validation dataset")

