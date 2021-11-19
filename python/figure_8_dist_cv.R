library(tidyverse)
library(jsonlite)
library(pROC)
suppressPackageStartupMessages(library(gridExtra))

cv<-data.frame()

models=c('r152','r101','r50','r34','r18')

# all these are pretrained
# we are using only results based on the best loss (i.e lowest MSE)
for(lr_ in c('0.0001')) {
    pfx=paste0('dist/lr_',lr_,'_pre/')
    for(v in models){
        for(r in c('_ref','')) {
            for(f in seq(0,7)) {
                fn=paste0(pfx,'model_',v,r,'/log_',f,'_8.json')
                if(file.exists(fn)) {
                rrr=fromJSON(fn)

                #final<-as.data.frame(rrr$testing_final$details)
                best<-as.data.frame(rrr$testing_best_loss$details)
                # calculate AUC, TODO: calculate ACC?
                #r_final<-roc(final$labels, final$preds, stratified=F, auc=T)
                r_best<-roc(best$labels, best$preds, stratified=F, auc=T)

                #as.data.frame(rrr$testing_final$summary) %>% mutate(select_kind='final',auc=r_final$auc),

                fold <- as.data.frame(rrr$testing_best_loss$summary) %>% 
                        mutate(select_kind='loss', auc=as.numeric(r_best$auc)) %>%
                        mutate(fold=f,ref=(r=="_ref"),model=v,lr=lr_, pre=T)

                print(fn)
                print(head(cv))
                print(head(fold))

                cv<-bind_rows(cv,fold)
                }
            }
        }
    }
}
cv<-cv%>%mutate(model=factor(model, levels=models))


ccv<-cv %>% mutate(rmse=sqrt(loss)) %>% 
    mutate( ref_=factor(ref, levels=c(FALSE, TRUE), labels=c('No Reference','With Reference')))

tccv<-ccv %>% group_by(model, ref_, pre, lr) %>% 
  summarize( rmse=median(rmse), rmse_lab=signif(rmse,4),
             auc=median(auc),auc_lab=signif(auc,4)) %>% ungroup()

# highlite
#tccv<-tccv %>% mutate( highlite=(score==best_tnr))


#    scale_colour_manual(labels=c(F,T), values=c('black','red'))+
#    theme(axis.text.x=element_text(angle=45,vjust=0.3))+

p1<-ggplot(ccv,aes(y=rmse, x=model))+
    theme_bw(base_size = 28)+
    geom_violin(trim = FALSE,alpha=0.8)+
    stat_summary(
        fun.data = "median_hilow", 
        geom = "pointrange", color = "black"
        )+
    facet_grid(ref_~.)+ylab('')+
    geom_text(data=tccv, aes(label=rmse_lab, x=model, y=rmse), 
              show.legend = F, size=6, nudge_y=0.2)+
    ggtitle("Root mean square error (mm)")

p2<-ggplot(ccv,aes(y=auc, x=model))+
    theme_bw(base_size = 28)+
    geom_violin(trim = FALSE,alpha=0.8)+
    stat_summary(
        fun.data = "median_hilow", 
        geom = "pointrange", color = "black"
        )+
    facet_grid(ref_~.)+ylab('')+
    geom_text(data=tccv, aes(label=auc_lab, x=model, y=auc), 
              show.legend = F, size=6, nudge_y=0.005)+
    ggtitle("Area under the curve")

png("Figure_8_DARQ_Dist_CV_performance_10epochs_all_models.png", 
    width=20, height=10, res=200, units = "in", 
    pointsize = 12, type='cairo', antialias = "default")

grid.arrange(p1,p2,nrow=1)

