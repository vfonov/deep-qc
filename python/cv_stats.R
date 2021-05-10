library(tidyverse)
library(jsonlite)

cv<-data.frame()

# all these are pretrained
for(v in c('r18')){
    for(r in c('_ref','')) {
        for(f in seq(0,7)) {
            rrr=fromJSON(paste0('model_',v,r,'/log_',f,'_8.json'))

            fold <- bind_rows(
                as.data.frame(rrr$testing_final$summary)   %>%mutate(select_kind='final'),
                as.data.frame(rrr$testing_best_acc$summary)%>%mutate(select_kind='acc'),
                as.data.frame(rrr$testing_best_auc$summary)%>%mutate(select_kind='auc'),
                as.data.frame(rrr$testing_best_tpr$summary)%>%mutate(select_kind='tpr'),
                as.data.frame(rrr$testing_best_tnr$summary)%>%mutate(select_kind='tnr')) %>%
                mutate(fold=f,ref=(r=="_ref"),model=v)
            cv<-bind_rows(cv,fold)
        }
    }
}

# all these are pretrained
cv<-cv %>% mutate(pre=T, select_kind=factor(select_kind,levels=c('final', 'auc', 'acc', 'tpr', 'tnr') ))

ccv<-cv %>% gather(`acc`,`tpr`,`tnr`,`auc`, key='measure', value='score')
tccv<-ccv %>% group_by(measure, model, ref, select_kind, pre) %>% 
  summarize( score=median(score), score_lab=signif(score,4)) %>%ungroup()

# select the best TNR
best_tnr<-max( (tccv%>%filter(measure=='tnr'))$score)

# highlite
tccv<-tccv %>% mutate( highlite=(score==best_tnr)&(measure=='tnr') )

#     geom_hline(data=tccv,aes(x=measure,yintercept=score))+


png("DARQ_CV_performance_10epochs_resnet18.png", width=2000, height=1000)

#    theme(axis.text.x=element_text(angle=45,vjust=0.3))+
ggplot(ccv,aes(y=score,x=select_kind))+
    theme_bw(base_size = 28)+
    geom_boxplot()+
    facet_wrap(ref~measure, labeller='label_both',ncol=4)+
    geom_text(data=tccv, aes(label=score_lab, x=select_kind, y=score,color=highlite),show.legend = F,size=6,nudge_y=0.03)+
    scale_colour_manual(labels=c(F,T),values=c('black','red'))+
    ggtitle('Model: Resnet-18 pretrained on ImageNet with and without reference')
