##########
#
# Show statistics on one CV fold
#
#
#
##########
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gridExtra))
library(pROC)


online<-data.frame()
final_summary<-data.frame()
final_detail<-data.frame()

models=c('r152','r101','r50','r34','r18')

for(m in models)
{
# validation stats
    r<-fromJSON(paste0('pre_distance/model_',m,'_ref_long/log_0_8.json'))
    val<-r$validation
    # convert minibatches to fractional epoch
    sc<-val%>%summarize(epoch=max(epoch),minibatch=max(ctr))%>%transmute(sc=epoch/minibatch)

    val<-val%>%mutate(f_epoch=ctr*sc$sc)

    val$model=m
    online<-bind_rows(online,val)

    final_<-as.data.frame(r$testing_final$summary) %>% mutate(model=m,f_epoch=max(val$f_epoch))
    final_summary<-bind_rows(final_summary,final_)

    final_detail<-bind_rows(final_detail,as.data.frame(r$testing_final$detail) %>% mutate(model=m))
}

online<-online %>%  mutate(model=factor(model,levels=models))
final_summary<-final_summary %>%  mutate(model=factor(model,levels=models))
final_detail<-final_detail %>%  mutate(model=factor(model,levels=models))

online_m <- online %>% gather(`acc`,`auc`,`tpr`,`tnr`, key='measure', value='score') %>%
    mutate(measure=factor(measure,levels=c('acc','auc','tpr','tnr'),
                   labels=c('Accuracy','Area under ROC curve','True positive rate','True negative rate')) )

# final_m<-final_summary %>% gather(`acc`,`auc`,`tpr`,`tnr`,key='measure', value='score')
#
#    
p1<-ggplot(online_m,aes(y=score, x=f_epoch, color=model))+
    theme_bw(base_size = 16)+
    facet_wrap(~measure,ncol=2)+
    geom_smooth(se = FALSE)+
    xlab('Epoch')+
    ggtitle("Online validation (balanced)")

# geom_point(data=final_m,aes(y=score, x=f_epoch, color=model),size=5,shape=4,stroke=1,show.legend=F)+


auc<-data.frame()
auc_s<-data.frame()
models_<-c()

for(s in models) {
    ss<-final_detail%>%filter(model==s)
    #pr<-pr.curve(scores.class0 = ss$labels , weights.class0 = ss$scores,curve=T)

    #apr<-bind_rows(apr, data.frame(pr$curve) %>% mutate(optimize=paste(s,format(pr$auc.integral,digits=2) ) ))
    #apr_s<-bind_rows(apr_s, data.frame(auc=pr$auc.integral, optimize=s))


    r<-roc(ss$labels, ss$scores, stratified=F,auc=T) # 

   #r_ci<-ci(r,of='thresholds',progress="none",parallel=T)
   #r_ci2<-ci(r)
    a<-paste(s,'auc:',format(r$auc,digits=3))

    models_<-c(models_,a)

    auc<-bind_rows(auc,data.frame(
                         tpr=r$sensitivities,
                         fpr=1.0-r$specificities
                        ) %>% mutate(model=a)
                   )

    auc_s<-bind_rows(auc_s, data.frame(auc=r$auc, model=s))
}
auc<-auc %>%  mutate(model=factor(model,levels=models_))
auc_s<-auc_s %>%  mutate(model=factor(model,levels=models))


print(auc_s)

#png("online_validation_final.png", width=10, height=10, res=200, units = "in", pointsize = 12, type='cairo', antialias = "default")
p2<-ggplot(data=auc,aes(y=tpr,x=fpr, color=model))+
  theme_bw(base_size = 14)+
  theme(
    legend.position = c(.95, .50),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6)
    )+
  geom_line()+
  coord_fixed()+
  geom_abline(slope=1.0,intercept=0.0, lty=2, col='red')+
  ggtitle("ROC curve on testing dataset (balanced)")



png("Figure_4_online_validation_testing.png", width=20, height=10, res=200, units = "in", pointsize = 12, type='cairo', antialias = "default")

grid.arrange(p1,p2,nrow=1,widths = c(2,1))

