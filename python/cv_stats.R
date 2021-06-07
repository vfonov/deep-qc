library(tidyverse)
library(jsonlite)

cv<-data.frame()

models=c('r152','r101','r50','r34','r18')

# all these are pretrained
for(lr_ in c('0.0001')) {
    pfx=paste0('lr_',lr_,'_pre/')
    for(v in models){
        for(r in c('_ref','')) {
            for(f in seq(0,7)) {
                fn=paste0(pfx,'model_',v,r,'/log_',f,'_8.json')
                if(file.exists(fn)) {
                rrr=fromJSON(fn)

                fold <- bind_rows(
                    as.data.frame(rrr$testing_final$summary)   %>%mutate(select_kind='final'),
                    as.data.frame(rrr$testing_best_acc$summary)%>%mutate(select_kind='acc'),
                    as.data.frame(rrr$testing_best_auc$summary)%>%mutate(select_kind='auc'),
                    as.data.frame(rrr$testing_best_tpr$summary)%>%mutate(select_kind='tpr'),
                    as.data.frame(rrr$testing_best_tnr$summary)%>%mutate(select_kind='tnr')) %>%
                    mutate(fold=f,ref=(r=="_ref"),model=v,lr=lr_, pre=T)
                cv<-bind_rows(cv,fold)
                }
            }
        }
    }
}
cv<-cv%>%mutate(model=factor(model, levels=models))

# HACK: all these are pretrained for now


cv_<-cv %>% filter(model=='r18')
# all these are pretrained


cv_<-cv_ %>% mutate( 
    select_kind=factor(select_kind, levels=c('final', 'auc', 'acc', 'tpr', 'tnr') ),
    lr=as.factor(lr))

ccv<-cv_ %>% gather(`acc`,`tpr`,`tnr`,`auc`, key='measure', value='score') %>% 
         mutate(measure=factor(measure,levels=c('acc','auc','tpr','tnr'),
                labels=c('Accuracy','Area under ROC curve',
                         'True positive rate',
                         'True negative rate')) )

tccv<-ccv %>% group_by(measure, model, ref, select_kind, pre, lr) %>% 
  summarize( score=median(score), score_lab=signif(score,4)) %>%ungroup()

# select the best TNR
best_tnr<-max( (tccv%>%filter(measure=='True negative rate'))$score)

# highlite
tccv<-tccv %>% mutate( highlite=(score==best_tnr)&(measure=='True negative rate') )

#     geom_hline(data=tccv,aes(x=measure,yintercept=score))+

png("DARQ_CV_performance_10epochs_r18.png", 
    width=20, height=10, res=200, units = "in", 
    pointsize = 12, type='cairo', antialias = "default")

#    theme(axis.text.x=element_text(angle=45,vjust=0.3))+
ggplot(ccv,aes(y=score,x=select_kind))+
    theme_bw(base_size = 28)+
    geom_boxplot()+
    facet_wrap(.~measure,ncol=2)+
    geom_text(data=tccv, aes(label=score_lab, x=select_kind, y=score,color=highlite),show.legend = F,size=6,nudge_y=0.03)+
    scale_colour_manual(labels=c(F,T),values=c('black','red'))+
    ggtitle('DARQ ResNet18 pretrained on ImageNet with and without reference, all measurements')

# focus only on TNR
ccv<-cv %>% gather(`acc`,`tpr`,`tnr`,`auc`, key='measure', value='score')

ccv<-ccv %>% filter(measure=='tnr')
tccv<-ccv %>% group_by(model, ref, select_kind, pre, lr) %>% 
  summarize( score=median(score), score_lab=signif(score,4)) %>%ungroup()

# select the best TNR
best_tnr<-max(tccv$score)

# highlite
tccv<-tccv %>% mutate( highlite=(score==best_tnr))

png("DARQ_CV_performance_10epochs_tnr.png", 
    width=20, height=10, res=200, units = "in", 
    pointsize = 12, type='cairo', antialias = "default")

#    theme(axis.text.x=element_text(angle=45,vjust=0.3))+
ggplot(ccv,aes(y=score, x=select_kind))+
    theme_bw(base_size = 28)+
    geom_boxplot()+
    facet_wrap(.~model, labeller='label_both',ncol=2)+
    geom_text(data=tccv, aes(label=score_lab, x=select_kind, y=score,color=highlite),show.legend = F,size=6,nudge_y=0.03)+
    scale_colour_manual(labels=c(F,T), values=c('black','red'))+
    ggtitle('DARQ 8-fold CV, TNR')
