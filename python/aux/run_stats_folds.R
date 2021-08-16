library(getopt)
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gridExtra))
#suppressPackageStartupMessages(library(PRROC))
library(pROC)
#library(doParallel)
#registerDoParallel(cl <- makeCluster(getOption("mc.cores", 16)))

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


inp=opt$input
cv<-data.frame()
details<-data.frame()

for(j in Sys.glob(file.path(inp,'log_*_*.json'))) {
    print(j)
    rrr=fromJSON(j)

    fold<-bind_rows(
        as.data.frame(rrr$testing_final$summary)   %>%mutate(optimize='final'),
        as.data.frame(rrr$testing_best_acc$summary)%>%mutate(optimize='acc'),
        as.data.frame(rrr$testing_best_auc$summary)%>%mutate(optimize='auc'),
        as.data.frame(rrr$testing_best_tpr$summary)%>%mutate(optimize='tpr'),
        as.data.frame(rrr$testing_best_tnr$summary)%>%mutate(optimize='tnr'))
    cv<-bind_rows(cv,fold)

    fold_details <- bind_rows(
        as.data.frame(rrr$testing_final$details)   %>%mutate(optimize='final'),
        as.data.frame(rrr$testing_best_acc$details)%>%mutate(optimize='acc'),
        as.data.frame(rrr$testing_best_auc$details)%>%mutate(optimize='auc'),
        as.data.frame(rrr$testing_best_tpr$details)%>%mutate(optimize='tpr'),
        as.data.frame(rrr$testing_best_tnr$details)%>%mutate(optimize='tnr')) %>%
        mutate(optimize=as.factor(optimize), labels=as.integer(labels))

    details<-bind_rows(details,fold_details)

    print(details%>%summarize(n=n()))
}
details<-details%>%mutate(optimize=factor(optimize, levels=c('final', 'auc', 'acc', 'tpr', 'tnr') ))
cv<-cv %>% mutate(optimize=factor(optimize, levels=c('final', 'auc', 'acc', 'tpr', 'tnr') ))

ccv<-cv %>% gather(`acc`,`tpr`,`tnr`,`auc`, key='measure', value='score')
tccv<-ccv %>% group_by(measure, optimize) %>% 
  summarize( score=median(score), score_lab=signif(score,4))%>%ungroup()

# select the best TNR
best_tnr<-max( (tccv%>%filter(measure=='tnr'))$score)

# highlite
tccv<-tccv %>% mutate( highlite=(score==best_tnr)&(measure=='tnr') )

#     geom_hline(data=tccv,aes(x=measure,yintercept=score))+


g1<-ggplot(ccv,aes(y=score,x=optimize))+
    theme_bw(base_size = 14)+
    theme(axis.text.x = element_text(angle = 45))+
    geom_boxplot()+
    facet_wrap(.~measure, labeller='label_both',ncol=8)+
    geom_text(data=tccv, aes(label=score_lab, x=optimize, y=score, color=highlite), 
               show.legend = F, size=2, nudge_y=0.003)+
    scale_colour_manual(labels=c(F,T), values=c('black','red'))

#apr<-data.frame()
#apr_s<-data.frame()


auc<-data.frame()
auc_s<-data.frame()


for(s in levels(details$optimize)) {
    ss<-details%>%filter(optimize==s)
    #pr<-pr.curve(scores.class0 = ss$labels , weights.class0 = ss$scores,curve=T)

    #apr<-bind_rows(apr, data.frame(pr$curve) %>% mutate(optimize=paste(s,format(pr$auc.integral,digits=2) ) ))
    #apr_s<-bind_rows(apr_s, data.frame(auc=pr$auc.integral, optimize=s))


   r<-roc(ss$labels, ss$scores, stratified=F,auc=T) # 

   #r_ci<-ci(r,of='thresholds',progress="none",parallel=T)
   #r_ci2<-ci(r)
   
   auc<-bind_rows(auc,data.frame(
                         tpr=r$sensitivities,
                         fpr=1.0-r$specificities
                         ) %>% mutate(optimize=paste(s,'auc:',format(r$auc,digits=3))
                        ))

    auc_s<-bind_rows(auc_s, data.frame(auc=r$auc, optimize=s))
}

print(auc_s)

# g2<-ggplot(apr,aes(x=X1,y=X2, color=optimize))+
#     theme_bw(base_size = 14)+
#     geom_line()+
#     coord_fixed()+
#     xlim(c(0,1))+ylim(c(0,1))+
#     geom_abline(slope=1.0,intercept=0.0,lty=2,col='red')+
#     labs(x="Recall", y="Precision")


g3<-ggplot(data=auc,aes(y=tpr,x=fpr, color=optimize))+
  theme_bw(base_size = 14)+
  geom_line()+
  coord_fixed()+
  geom_abline(slope=1.0,intercept=0.0, lty=2, col='red')

png(opt$output, width=15, height=7, res=200, units = "in", pointsize = 12, type='cairo', antialias = "default")
grid.arrange(g1,g3,nrow=1)

# PRINT false positives
ss<-details %>% filter(optimize=='final')

# splitting into variant,cohort,subject,visit
sss<-str_split_fixed(ss$ids,":",4) %>% as.data.frame() %>%
 rename( variant=V1, cohort=V2, subject=V3, visit=V4)

ss<-bind_cols(ss,sss)%>%select(variant, cohort, subject, visit, scores, labels)


# select worst offenders
worst_fpr<-ss%>%filter(labels==0,scores>0.99)%>%arrange(-scores)

print("False positives for final")
print(head(worst_fpr))

write_csv(worst_fpr,'worst_fpr.csv')

print("False negatives for final")
worst_fnr<-ss%>%filter(labels==1,scores<0.01)%>%arrange(scores) 

# select worst offenders
print(head(worst_fnr))

write_csv(worst_fnr,'worst_fnr.csv')
