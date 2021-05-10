library(tidyverse)
library(jsonlite)

cv<-data.frame()

for(v in c('r18','sq101')){
    for(r in c('_ref','')) {
        for(f in seq(0,7)) {
            rrr=fromJSON(paste0('model_',v,r,'/log_',f,'_8.json'))

            fold <- bind_rows(
                as.data.frame(rrr$testing_best_acc$summary)%>%mutate(select_kind='best_acc'),
                as.data.frame(rrr$testing_best_auc$summary)%>%mutate(select_kind='best_auc'),
                as.data.frame(rrr$testing_best_tpr$summary)%>%mutate(select_kind='best_tpr'),
                as.data.frame(rrr$testing_best_tnr$summary)%>%mutate(select_kind='best_tnr')) %>%
                mutate(fold=f,ref=(r=="_ref"),model=v)
            cv<-bind_rows(cv,fold)
        }
    }
}

ccv<-cv %>% gather(`acc`,`tpr`,`tnr`,`auc`,key='measure',value='score')
tccv<-ccv %>% group_by(measure,model,ref,select_kind) %>% 
 summarize(score=median(score),score_lab=signif(score,4))

#     geom_hline(data=tccv,aes(x=measure,yintercept=score))+


png("DARQ_CV_performance_20epochs.png", width=2000, height=2000)

ggplot(ccv,aes(y=score,x=select_kind))+
    theme_bw(base_size = 25)+
    theme(axis.text.x=element_text(angle=45,vjust=0.3))+
    geom_boxplot()+
    facet_wrap(measure~ref+model, labeller='label_both')+
    geom_text(data=tccv,aes(label=score_lab,x=select_kind,y=score),size=5,nudge_y=0.03)+
    xtitle('')
