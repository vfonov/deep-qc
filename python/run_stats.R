library(getopt)
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gridExtra))
library(PRROC)

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

cv<-data.frame()

rrr=fromJSON(opt$input)

fold<-bind_rows(
    as.data.frame(rrr$testing_final$summary)   %>%mutate(select_kind='final'),
    as.data.frame(rrr$testing_best_acc$summary)%>%mutate(select_kind='acc'),
    as.data.frame(rrr$testing_best_auc$summary)%>%mutate(select_kind='auc'),
    as.data.frame(rrr$testing_best_tpr$summary)%>%mutate(select_kind='tpr'),
    as.data.frame(rrr$testing_best_tnr$summary)%>%mutate(select_kind='tnr'))

cv<-bind_rows(cv,fold)

cv<-cv %>% mutate(select_kind=factor(select_kind, levels=c('final', 'auc', 'acc', 'tpr', 'tnr') ))

ccv<-cv %>% gather(`acc`,`tpr`,`tnr`,`auc`, key='measure', value='score')
tccv<-ccv %>% group_by(measure, select_kind) %>% 
  summarize( score=median(score), score_lab=signif(score,4))%>%ungroup()

# select the best TNR
best_tnr<-max( (tccv%>%filter(measure=='tnr'))$score)

# highlite
tccv<-tccv %>% mutate( highlite=(score==best_tnr)&(measure=='tnr') )

#     geom_hline(data=tccv,aes(x=measure,yintercept=score))+

png(opt$output, width=20, height=10, res=200, units = "in", pointsize = 12, type='cairo', antialias = "default")

g1<-ggplot(ccv,aes(y=score,x=select_kind))+
    theme_bw(base_size = 16)+
    geom_boxplot()+
    facet_wrap(.~measure, labeller='label_both',ncol=8)+
    geom_text(data=tccv, aes(label=score_lab, x=select_kind, y=score,color=highlite),show.legend = F,size=3,nudge_y=0.03)+
    scale_colour_manual(labels=c(F,T),values=c('black','red'))+
    ggtitle("Performance on testing dataset")

# long performance
long<-rrr$validation

long_m<-long%>%gather(`acc`,`prec`,`auc`,`tpr`,`tnr`,key='measure', value='score')

g2<-ggplot(long_m,aes(y=score, x=ctr,color=measure,fill=measure))+
    theme_bw(base_size = 16)+
    geom_smooth()+xlab('minibatch')+
    ggtitle("Performance on validation dataset")



# precision-recall curves
e <- bind_rows(
    as.data.frame(rrr$testing_final$details)   %>%mutate(select_kind='final'),
    as.data.frame(rrr$testing_best_acc$details)%>%mutate(select_kind='acc'),
    as.data.frame(rrr$testing_best_auc$details)%>%mutate(select_kind='auc'),
    as.data.frame(rrr$testing_best_tpr$details)%>%mutate(select_kind='tpr'),
    as.data.frame(rrr$testing_best_tnr$details)%>%mutate(select_kind='tnr') ) %>%
    mutate(select_kind=as.factor(select_kind),labels=as.integer(labels))

apr=data.frame()
apr_s=data.frame()

for(s in levels(e$select_kind)) {
    ss<-e%>%filter(select_kind==s)
    pr<-pr.curve(scores.class0 = ss$labels , weights.class0 = ss$scores,curve=T)
    apr<-bind_rows(apr,data.frame(pr$curve)%>%mutate(select_kind=paste(s,format(pr$auc.integral,digits=3) ) ))

    apr_s<-bind_rows(apr_s,data.frame(auc=pr$auc.integral,select_kind=s))
}

print(apr_s)

g3<-ggplot(apr,aes(x=X1,y=X2, color=select_kind))+
    theme_bw(base_size = 16)+
    geom_line()+
    labs(x="Recall", y="Precision")

grid.arrange(g1,g2,g3,nrow=1)



